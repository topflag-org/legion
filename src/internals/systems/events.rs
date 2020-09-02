//! Contains types related to defining shared events which can be accessed inside systems.
//!
//! Use events to share persistent data between systems or to provide a system with state
//! external to entities.

use crate::internals::{
    hash::ComponentTypeIdHasher,
    query::view::{read::Read, write::Write, ReadOnly},
};
use downcast_rs::{impl_downcast, Downcast};
use std::{
    any::TypeId,
    cell::UnsafeCell,
    collections::{hash_map::Entry, HashMap},
    fmt::{Display, Formatter},
    hash::{BuildHasherDefault, Hasher},
    marker::PhantomData,
    ops::{Deref, DerefMut},
    sync::atomic::AtomicIsize,
};

/// Unique ID for a event.
#[derive(Copy, Clone, Debug, Eq, PartialOrd, Ord)]
pub struct EventTypeId {
    type_id: TypeId,
    #[cfg(debug_assertions)]
    name: &'static str,
}

impl EventTypeId {
    /// Returns the event type ID of the given event type.
    pub fn of<T: Event>() -> Self {
        Self {
            type_id: TypeId::of::<T>(),
            #[cfg(debug_assertions)]
            name: std::any::type_name::<T>(),
        }
    }
}

impl std::hash::Hash for EventTypeId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.type_id.hash(state);
    }
}

impl PartialEq for EventTypeId {
    fn eq(&self, other: &Self) -> bool {
        self.type_id.eq(&other.type_id)
    }
}

impl Display for EventTypeId {
    #[cfg(debug_assertions)]
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }

    #[cfg(not(debug_assertions))]
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.type_id)
    }
}

/// Blanket trait for event types.
pub trait Event: 'static + Downcast {}
impl<T> Event for T where T: 'static {}
impl_downcast!(Event);

/// Trait which is implemented for tuples of events and singular events. This abstracts
/// fetching events to allow for ergonomic fetching.
///
/// # Example:
/// ```
///
/// struct TypeA(usize);
/// struct TypeB(usize);
///
/// # use legion::*;
/// # use legion::systems::EventSet;
/// let mut events = Events::default();
/// events.insert(TypeA(55));
/// events.insert(TypeB(12));
///
/// {
///     let (a, mut b) = <(Read<TypeA>, Write<TypeB>)>::fetch_mut(&mut events);
///     assert_ne!(a.0, b.0);
///     b.0 = a.0;
/// }
///
/// {
///     let (a, b) = <(Read<TypeA>, Read<TypeB>)>::fetch(&events);
///     assert_eq!(a.0, b.0);
/// }
///
/// ```
pub trait EventSet<'a> {
    /// The event reference returned during a fetch.
    type Result: 'a;

    /// Fetches all defined events, without checking mutability.
    ///
    /// # Safety
    /// It is up to the end user to validate proper mutability rules across the events being accessed.
    unsafe fn fetch_unchecked(events: &'a UnsafeEvents) -> Self::Result;

    /// Fetches all defined events.
    fn fetch_mut(events: &'a mut Events) -> Self::Result {
        // safe because mutable borrow ensures exclusivity
        unsafe { Self::fetch_unchecked(&events.internal) }
    }

    /// Fetches all defined events.
    fn fetch(events: &'a Events) -> Self::Result
    where
        Self: ReadOnly,
    {
        unsafe { Self::fetch_unchecked(&events.internal) }
    }
}

impl<'a> EventSet<'a> for () {
    type Result = ();

    unsafe fn fetch_unchecked(_: &UnsafeEvents) -> Self::Result {}
}

impl<'a, T: Event> EventSet<'a> for Read<T> {
    type Result = Fetch<'a, T>;

    unsafe fn fetch_unchecked(events: &'a UnsafeEvents) -> Self::Result {
        let type_id = &EventTypeId::of::<T>();
        events.get(&type_id).unwrap().get::<T>().unwrap()
    }
}

impl<'a, T: Event> EventSet<'a> for Write<T> {
    type Result = FetchMut<'a, T>;

    unsafe fn fetch_unchecked(events: &'a UnsafeEvents) -> Self::Result {
        let type_id = &EventTypeId::of::<T>();
        events.get(&type_id).unwrap().get_mut::<T>().unwrap()
    }
}

macro_rules! event_tuple {
    ($head_ty:ident) => {
        impl_event_tuple!($head_ty);
    };
    ($head_ty:ident, $( $tail_ty:ident ),*) => (
        impl_event_tuple!($head_ty, $( $tail_ty ),*);
        event_tuple!($( $tail_ty ),*);
    );
}

macro_rules! impl_event_tuple {
    ( $( $ty: ident ),* ) => {
        #[allow(unused_parens, non_snake_case)]
        impl<'a, $( $ty: EventSet<'a> ),*> EventSet<'a> for ($( $ty, )*)
        {
            type Result = ($( $ty::Result, )*);

            unsafe fn fetch_unchecked(events: &'a UnsafeEvents) -> Self::Result {
                ($( $ty::fetch_unchecked(events), )*)
            }
        }
    };
}

#[cfg(feature = "extended-tuple-impls")]
event_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z);

#[cfg(not(feature = "extended-tuple-impls"))]
event_tuple!(A, B, C, D, E, F, G, H);

/// Ergonomic wrapper type which contains a `Ref` type.
pub struct Fetch<'a, T: Event> {
    state: &'a AtomicIsize,
    inner: &'a T,
}

impl<'a, T: Event> Deref for Fetch<'a, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.inner
    }
}

impl<'a, T: 'a + Event + std::fmt::Debug> std::fmt::Debug for Fetch<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.deref())
    }
}

impl<'a, T: Event> Drop for Fetch<'a, T> {
    fn drop(&mut self) {
        self.state
            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Ergonomic wrapper type which contains a `RefMut` type.
pub struct FetchMut<'a, T: Event> {
    state: &'a AtomicIsize,
    inner: &'a mut T,
}

impl<'a, T: 'a + Event> Deref for FetchMut<'a, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &*self.inner
    }
}

impl<'a, T: 'a + Event> DerefMut for FetchMut<'a, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        &mut *self.inner
    }
}

impl<'a, T: 'a + Event + std::fmt::Debug> std::fmt::Debug for FetchMut<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.deref())
    }
}

impl<'a, T: Event> Drop for FetchMut<'a, T> {
    fn drop(&mut self) {
        self.state
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

pub struct EventCell {
    data: UnsafeCell<Box<dyn Event>>,
    borrow_state: AtomicIsize,
}

impl EventCell {
    fn new(event: Box<dyn Event>) -> Self {
        Self {
            data: UnsafeCell::new(event),
            borrow_state: AtomicIsize::new(0),
        }
    }

    fn into_inner(self) -> Box<dyn Event> {
        self.data.into_inner()
    }

    /// # Safety
    /// Types which are !Sync should only be retrieved on the thread which owns the event
    /// collection.
    pub unsafe fn get<T: Event>(&self) -> Option<Fetch<'_, T>> {
        loop {
            let read = self.borrow_state.load(std::sync::atomic::Ordering::SeqCst);
            if read < 0 {
                panic!(
                    "event already borrowed as mutable: {}",
                    std::any::type_name::<T>()
                );
            }

            if self.borrow_state.compare_and_swap(
                read,
                read + 1,
                std::sync::atomic::Ordering::SeqCst,
            ) == read
            {
                break;
            }
        }

        let event = self.data.get().as_ref().and_then(|r| r.downcast_ref::<T>());
        if let Some(event) = event {
            Some(Fetch {
                state: &self.borrow_state,
                inner: event,
            })
        } else {
            self.borrow_state
                .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            None
        }
    }

    /// # Safety
    /// Types which are !Send should only be retrieved on the thread which owns the event
    /// collection.
    pub unsafe fn get_mut<T: Event>(&self) -> Option<FetchMut<'_, T>> {
        let borrowed =
            self.borrow_state
                .compare_and_swap(0, -1, std::sync::atomic::Ordering::SeqCst);
        match borrowed {
            0 => {
                let event = self.data.get().as_mut().and_then(|r| r.downcast_mut::<T>());
                if let Some(event) = event {
                    Some(FetchMut {
                        state: &self.borrow_state,
                        inner: event,
                    })
                } else {
                    self.borrow_state
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    None
                }
            }
            x if x < 0 => panic!(
                "event already borrowed as mutable: {}",
                std::any::type_name::<T>()
            ),
            _ => panic!(
                "event already borrowed as immutable: {}",
                std::any::type_name::<T>()
            ),
        }
    }
}

/// A container for events which performs runtime borrow checking
/// but _does not_ ensure that `!Sync` events aren't accessed across threads.
#[derive(Default)]
pub struct UnsafeEvents {
    map: HashMap<EventTypeId, EventCell, BuildHasherDefault<ComponentTypeIdHasher>>,
}

unsafe impl Send for UnsafeEvents {}
unsafe impl Sync for UnsafeEvents {}

impl UnsafeEvents {
    fn contains(&self, type_id: &EventTypeId) -> bool {
        self.map.contains_key(type_id)
    }

    /// # Safety
    /// Events which are `!Sync` or `!Send` must be retrieved or inserted only on the main thread.
    unsafe fn entry(&mut self, type_id: EventTypeId) -> Entry<EventTypeId, EventCell> {
        self.map.entry(type_id)
    }

    /// # Safety
    /// Events which are `!Send` must be retrieved or inserted only on the main thread.
    unsafe fn insert<T: Event>(&mut self, event: T) {
        self.map.insert(
            EventTypeId::of::<T>(),
            EventCell::new(Box::new(event)),
        );
    }

    /// # Safety
    /// Events which are `!Send` must be retrieved or inserted only on the main thread.
    unsafe fn remove(&mut self, type_id: &EventTypeId) -> Option<Box<dyn Event>> {
        self.map.remove(type_id).map(|cell| cell.into_inner())
    }

    fn get(&self, type_id: &EventTypeId) -> Option<&EventCell> {
        self.map.get(type_id)
    }

    /// # Safety
    /// Events which are `!Sync` must be retrieved or inserted only on the main thread.
    unsafe fn merge(&mut self, mut other: Self) {
        // Merge events, retaining our local ones but moving in any non-existant ones
        for event in other.map.drain() {
            self.map.entry(event.0).or_insert(event.1);
        }
    }
}

/// Events container. Shared events stored here can be retrieved in systems.
#[derive(Default)]
pub struct Events {
    internal: UnsafeEvents,
    // marker to make `Events` !Send and !Sync
    //_not_send_sync: PhantomData<*const u8>,
}

impl Events {
    pub(crate) fn internal(&self) -> &UnsafeEvents {
        &self.internal
    }

    /// Creates an accessor to events which are Send and Sync, which itself can be sent
    /// between threads.
    pub fn sync(&mut self) -> SyncEvents {
        SyncEvents {
            internal: &self.internal,
        }
    }

    /// Returns `true` if type `T` exists in the store. Otherwise, returns `false`.
    pub fn contains<T: Event>(&self) -> bool {
        self.internal.contains(&EventTypeId::of::<T>())
    }

    /// Inserts the instance of `T` into the store. If the type already exists, it will be silently
    /// overwritten. If you would like to retain the instance of the event that already exists,
    /// call `remove` first to retrieve it.
    pub fn insert<T: Event>(&mut self, value: T) {
        // safety:
        // this type is !Send and !Sync, and so can only be accessed from the thread which
        // owns the events collection
        unsafe {
            self.internal.insert(value);
        }
    }

    /// Removes the type `T` from this store if it exists.
    ///
    /// # Returns
    /// If the type `T` was stored, the inner instance of `T is returned. Otherwise, `None`.
    pub fn remove<T: Event>(&mut self) -> Option<T> {
        // safety:
        // this type is !Send and !Sync, and so can only be accessed from the thread which
        // owns the events collection
        unsafe {
            let event = self
                .internal
                .remove(&EventTypeId::of::<T>())?
                .downcast::<T>()
                .ok()?;
            Some(*event)
        }
    }

    /// Retrieve an immutable reference to  `T` from the store if it exists. Otherwise, return `None`.
    ///
    /// # Panics
    /// Panics if the event is already borrowed mutably.
    pub fn get<T: Event>(&self) -> Option<Fetch<'_, T>> {
        // safety:
        // this type is !Send and !Sync, and so can only be accessed from the thread which
        // owns the events collection
        let type_id = &EventTypeId::of::<T>();
        unsafe { self.internal.get(&type_id)?.get::<T>() }
    }

    /// Retrieve a mutable reference to  `T` from the store if it exists. Otherwise, return `None`.
    pub fn get_mut<T: Event>(&self) -> Option<FetchMut<'_, T>> {
        // safety:
        // this type is !Send and !Sync, and so can only be accessed from the thread which
        // owns the events collection
        let type_id = &EventTypeId::of::<T>();
        unsafe { self.internal.get(&type_id)?.get_mut::<T>() }
    }

    /// Attempts to retrieve an immutable reference to `T` from the store. If it does not exist,
    /// the closure `f` is called to construct the object and it is then inserted into the store.
    pub fn get_or_insert_with<T: Event, F: FnOnce() -> T>(&mut self, f: F) -> Fetch<'_, T> {
        // safety:
        // this type is !Send and !Sync, and so can only be accessed from the thread which
        // owns the events collection
        let type_id = EventTypeId::of::<T>();
        unsafe {
            self.internal
                .entry(type_id)
                .or_insert_with(|| EventCell::new(Box::new((f)())))
                .get()
                .unwrap()
        }
    }

    /// Attempts to retrieve a mutable reference to `T` from the store. If it does not exist,
    /// the closure `f` is called to construct the object and it is then inserted into the store.
    pub fn get_mut_or_insert_with<T: Event, F: FnOnce() -> T>(
        &mut self,
        f: F,
    ) -> FetchMut<'_, T> {
        // safety:
        // this type is !Send and !Sync, and so can only be accessed from the thread which
        // owns the events collection
        let type_id = EventTypeId::of::<T>();
        unsafe {
            self.internal
                .entry(type_id)
                .or_insert_with(|| EventCell::new(Box::new((f)())))
                .get_mut()
                .unwrap()
        }
    }

    /// Attempts to retrieve an immutable reference to `T` from the store. If it does not exist,
    /// the provided value is inserted and then a reference to it is returned.
    pub fn get_or_insert<T: Event>(&mut self, value: T) -> Fetch<'_, T> {
        self.get_or_insert_with(|| value)
    }

    /// Attempts to retrieve a mutable reference to `T` from the store. If it does not exist,
    /// the provided value is inserted and then a reference to it is returned.
    pub fn get_mut_or_insert<T: Event>(&mut self, value: T) -> FetchMut<'_, T> {
        self.get_mut_or_insert_with(|| value)
    }

    /// Attempts to retrieve an immutable reference to `T` from the store. If it does not exist,
    /// the default constructor for `T` is called.
    ///
    /// `T` must implement `Default` for this method.
    pub fn get_or_default<T: Event + Default>(&mut self) -> Fetch<'_, T> {
        self.get_or_insert_with(T::default)
    }

    /// Attempts to retrieve a mutable reference to `T` from the store. If it does not exist,
    /// the default constructor for `T` is called.
    ///
    /// `T` must implement `Default` for this method.
    pub fn get_mut_or_default<T: Event + Default>(&mut self) -> FetchMut<'_, T> {
        self.get_mut_or_insert_with(T::default)
    }

    /// Performs merging of two event storages, which occurs during a world merge.
    /// This merge will retain any already-existant events in the local world, while moving any
    /// new events from the source world into this one, consuming the events.
    pub fn merge(&mut self, other: Events) {
        // safety:
        // this type is !Send and !Sync, and so can only be accessed from the thread which
        // owns the events collection
        unsafe {
            self.internal.merge(other.internal);
        }
    }
}

/// A event collection which is `Send` and `Sync`, but which only allows access to events
/// which are `Sync`.
pub struct SyncEvents<'a> {
    internal: &'a UnsafeEvents,
}

impl<'a> SyncEvents<'a> {
    /// Retrieve an immutable reference to  `T` from the store if it exists. Otherwise, return `None`.
    ///
    /// # Panics
    /// Panics if the event is already borrowed mutably.
    pub fn get<T: Event + Sync>(&self) -> Option<Fetch<'_, T>> {
        // safety:
        // only events which are Sync can be accessed, and so are safe to access from any thread
        let type_id = &EventTypeId::of::<T>();
        unsafe { self.internal.get(&type_id)?.get::<T>() }
    }

    /// Retrieve a mutable reference to  `T` from the store if it exists. Otherwise, return `None`.
    pub fn get_mut<T: Event + Send>(&self) -> Option<FetchMut<'_, T>> {
        // safety:
        // only events which are Send can be accessed, and so are safe to access from any thread
        let type_id = &EventTypeId::of::<T>();
        unsafe { self.internal.get(&type_id)?.get_mut::<T>() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_read_write_test() {
        struct TestOne {
            value: String,
        }

        struct TestTwo {
            value: String,
        }

        struct NotSync {
            ptr: *const u8,
        }

        let mut events = Events::default();
        events.insert(TestOne {
            value: "one".to_string(),
        });

        events.insert(TestTwo {
            value: "two".to_string(),
        });

        events.insert(NotSync {
            ptr: std::ptr::null(),
        });

        assert_eq!(events.get::<TestOne>().unwrap().value, "one");
        assert_eq!(events.get::<TestTwo>().unwrap().value, "two");
        assert_eq!(events.get::<NotSync>().unwrap().ptr, std::ptr::null());

        // test re-ownership
        let owned = events.remove::<TestTwo>();
        assert_eq!(owned.unwrap().value, "two");
    }
}
