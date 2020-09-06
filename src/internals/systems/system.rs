//! Contains types related to the definition of systems.

use super::{
    command::CommandBuffer,
    resources::{Resource, ResourceSet, ResourceTypeId, UnsafeResources},
    events::{Event, EventSet, EventTypeId, UnsafeEvents},
    schedule::Runnable,
};
use crate::internals::{
    cons::{ConsAppend, ConsFlatten},
    permissions::Permissions,
    query::{
        filter::EntityFilter,
        view::{read::Read, write::Write, IntoView, View},
        Query,
    },
    storage::{
        archetype::ArchetypeIndex,
        component::{Component, ComponentTypeId},
    },
    subworld::{ArchetypeAccess, ComponentAccess, SubWorld},
    world::{World, WorldId},
};
use bit_set::BitSet;
use std::{any::{Any, TypeId}, borrow::Cow, collections::HashMap, error::Error, marker::PhantomData};
use tracing::{debug, info, span, Level};
use smallvec::SmallVec;

/// Result of the execution of a system.
/// Returns a SmallVec of events, or an error.
pub type SystemResult = Result<SmallVec<[Box<dyn Event>; 4]>, Box<dyn Error>>;
// TODO
//pub type SystemResult = Result<SmallVec<[(EventTypeId, Box<dyn Event>); 4]>, Box<dyn Error>>;

/// Provides an abstraction across tuples of queries for system closures.
pub trait QuerySet: Send + Sync {
    /// Evaluates the queries and records which archetypes they require access to into a bitset.
    fn filter_archetypes(&mut self, world: &World, archetypes: &mut BitSet);
}

macro_rules! queryset_tuple {
    ($head_ty:ident) => {
        impl_queryset_tuple!($head_ty);
    };
    ($head_ty:ident, $( $tail_ty:ident ),*) => (
        impl_queryset_tuple!($head_ty, $( $tail_ty ),*);
        queryset_tuple!($( $tail_ty ),*);
    );
}

macro_rules! impl_queryset_tuple {
    ($($ty: ident),*) => {
            #[allow(unused_parens, non_snake_case)]
            impl<$( $ty, )*> QuerySet for ($( $ty, )*)
            where
                $( $ty: QuerySet, )*
            {
                fn filter_archetypes(&mut self, world: &World, bitset: &mut BitSet) {
                    let ($($ty,)*) = self;

                    $( $ty.filter_archetypes(world, bitset); )*
                }
            }
    };
}

#[cfg(feature = "extended-tuple-impls")]
queryset_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z);

#[cfg(not(feature = "extended-tuple-impls"))]
queryset_tuple!(A, B, C, D, E, F, G, H);

impl QuerySet for () {
    fn filter_archetypes(&mut self, _: &World, _: &mut BitSet) {}
}

impl<AV, AF> QuerySet for Query<AV, AF>
where
    AV: IntoView + Send + Sync,
    AF: EntityFilter,
{
    fn filter_archetypes(&mut self, world: &World, bitset: &mut BitSet) {
        for &ArchetypeIndex(arch) in self.find_archetypes(world) {
            bitset.insert(arch as usize);
        }
    }
}

/// Structure describing the resource and component access conditions of the system.
#[derive(Debug, Clone)]
pub struct SystemAccess {
    resources: Permissions<ResourceTypeId>,
    components: Permissions<ComponentTypeId>,
}

/// A diagnostic identifier for a system.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SystemId {
    name: Cow<'static, str>,
    type_id: TypeId,
}

struct Unspecified;

impl std::fmt::Display for SystemId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl<T: Into<Cow<'static, str>>> From<T> for SystemId {
    fn from(name: T) -> SystemId {
        SystemId {
            name: name.into(),
            type_id: TypeId::of::<Unspecified>(),
        }
    }
}

struct ResourceMarker<T>(PhantomData<*const T>);
unsafe impl<T: Send> Send for ResourceMarker<T> {}
unsafe impl<T: Sync> Sync for ResourceMarker<T> {}

struct EventMarker<T>(PhantomData<*const T>);
unsafe impl<T: Send> Send for EventMarker<T> {}
unsafe impl<T: Sync> Sync for EventMarker<T> {}

/// The concrete type which contains the system closure provided by the user.  This struct should
/// not be constructed directly, and instead should be created using `SystemBuilder`.
///
/// Implements `Schedulable` which is consumable by the `StageExecutor`, executing the closure.
///
/// Also handles caching of archetype information, as well as maintaining the provided
/// information about what queries this system will run and, as a result, its data access.
pub struct System<E, R, Q, F> {
    name: Option<SystemId>,
    _events: EventMarker<E>,
    _resources: ResourceMarker<R>,
    queries: Q,
    run_fn: F,
    archetypes: ArchetypeAccess,
    access: SystemAccess,
    event_types: SmallVec<[EventTypeId; 2]>,
    command_buffer: HashMap<WorldId, CommandBuffer>,
}

impl<E, R, Q, F> Runnable for System<E, R, Q, F>
where
    E: for<'a> EventSet<'a>,
    R: for<'a> ResourceSet<'a>,
    Q: QuerySet,
    F: SystemFn<E, R, Q>,
{
    fn name(&self) -> Option<&SystemId> {
        self.name.as_ref()
    }

    fn events(&self) -> &[EventTypeId] {
        self.event_types.as_slice()
    }

    fn reads(&self) -> (&[ResourceTypeId], &[ComponentTypeId]) {
        (
            &self.access.resources.reads_only(),
            &self.access.components.reads_only(),
        )
    }

    fn writes(&self) -> (&[ResourceTypeId], &[ComponentTypeId]) {
        (
            &self.access.resources.writes(),
            &self.access.components.writes(),
        )
    }

    fn prepare(&mut self, world: &World) {
        if let ArchetypeAccess::Some(bitset) = &mut self.archetypes {
            self.queries.filter_archetypes(world, bitset);
        }
    }

    fn accesses_archetypes(&self) -> &ArchetypeAccess {
        &self.archetypes
    }

    fn command_buffer_mut(&mut self, world: WorldId) -> Option<&mut CommandBuffer> {
        self.command_buffer.get_mut(&world)
    }

    unsafe fn run_unsafe(&mut self, world: &World, events: &UnsafeEvents, resources: &UnsafeResources) -> SystemResult {
        let span = if let Some(name) = &self.name {
            span!(Level::INFO, "System", system = %name)
        } else {
            span!(Level::INFO, "System")
        };
        let _guard = span.enter();

        debug!("Initializing");

        // safety:
        // It is difficult to correctly communicate the lifetime of the resource fetch through to the system closure.
        // We are hacking this by passing the fetch with a static lifetime to its internal references.
        // This is sound because the fetch structs only provide access to the resource through reborrows on &self.
        // As the fetch struct is created on the stack here, and the resources it is holding onto is a parameter to this function,
        // we know for certain that the lifetime of the fetch struct (which constrains the lifetime of the resource the system sees)
        // must be shorter than the lifetime of the resource.
        let resources_static = &*(resources as *const UnsafeResources);
        let mut resources = R::fetch_unchecked(resources_static);

        let events_static = &*(events as *const UnsafeEvents);
        let events = E::fetch_unchecked(events_static);

        let queries = &mut self.queries;
        let component_access = ComponentAccess::Allow(Cow::Borrowed(&self.access.components));
        let mut world_shim =
            SubWorld::new_unchecked(world, component_access, self.archetypes.bitset());
        let cmd = self
            .command_buffer
            .entry(world.id())
            .or_insert_with(|| CommandBuffer::new(world));

        info!("Running");
        let borrow = &mut self.run_fn;
        borrow.run(cmd, &mut world_shim, &events, &mut resources, queries)
    }
}

/// A function which can provide the body of a system.
pub trait SystemFn<E: EventSet<'static>, R: ResourceSet<'static>, Q: QuerySet> {
    /// Runs the system body.
    fn run(
        &mut self,
        commands: &mut CommandBuffer,
        world: &mut SubWorld,
        events: &E::Result,
        resources: &mut R::Result,
        queries: &mut Q,
    ) -> SystemResult;
}

impl<E, F, R, Q> SystemFn<E, R, Q> for F
where
    E: EventSet<'static>,
    R: ResourceSet<'static>,
    Q: QuerySet,
    F: FnMut(&mut CommandBuffer, &mut SubWorld, &E::Result, &mut R::Result, &mut Q) -> SystemResult + 'static,
{
    fn run(
        &mut self,
        commands: &mut CommandBuffer,
        world: &mut SubWorld,
        events: &E::Result,
        resources: &mut R::Result,
        queries: &mut Q,
    ) -> SystemResult {
        (self)(commands, world, events, resources, queries)
    }
}

// This builder uses a Cons/Hlist implemented in cons.rs to generated the static query types
// for this system. Access types are instead stored and abstracted in the top level vec here
// so the underlying ResourceSet type functions from the queries don't need to allocate.
// Otherwise, this leads to excessive alloaction for every call to reads/writes

/// A low level builder for constructing systems.
/// ```rust
/// # use legion::*;
/// # #[derive(Copy, Clone, Debug, PartialEq)]
/// # struct Position;
/// # #[derive(Copy, Clone, Debug, PartialEq)]
/// # struct Velocity;
/// # #[derive(Copy, Clone, Debug, PartialEq)]
/// # struct Model;
/// #[derive(Copy, Clone, Debug, PartialEq)]
/// struct Static;
/// #[derive(Debug)]
/// struct TestResource {}
///
///  let mut system_one = SystemBuilder::new("TestSystem")
///            .read_resource::<TestResource>()
///            .with_query(<(Entity, Read<Position>, Read<Model>)>::query()
///                         .filter(!component::<Static>() | maybe_changed::<Position>()))
///            .build(move |commands, world, events, resource, queries| {
///                for (entity, pos, model) in queries.iter_mut(world) {
///
///                }
///            });
/// ```
pub struct SystemBuilder<Q = (), R = (), E = ()> {
    name: Option<SystemId>,
    queries: Q,
    resources: R,
    events: E,
    event_types: SmallVec<[EventTypeId;2]>,
    resource_access: Permissions<ResourceTypeId>,
    component_access: Permissions<ComponentTypeId>,
    access_all_archetypes: bool,
}

impl SystemBuilder<(), (), ()> {
    /// Create a new system builder to construct a new system.
    ///
    /// Please note, the `name` argument for this method is just for debugging and visualization
    /// purposes and is not logically used anywhere.
    pub fn new<T: Into<SystemId>>(name: T) -> Self {
        Self {
            name: Some(name.into()),
            ..Self::default()
        }
    }
}

impl Default for SystemBuilder<(), (), ()> {
    fn default() -> Self {
        Self {
            name: None,
            queries: (),
            resources: (),
            events: (),
            event_types: SmallVec::new(),
            resource_access: Permissions::default(),
            component_access: Permissions::default(),
            access_all_archetypes: false,
        }
    }
}

impl<Q, R, E> SystemBuilder<Q, R, E>
where
    Q: 'static + Send + ConsFlatten,
    R: 'static + Send + ConsFlatten,
    E: 'static + Send + ConsFlatten,
{
    /// Provides a name to the system being built.
    pub fn with_name(self, name: SystemId) -> Self {
        Self {
            name: Some(name),
            ..self
        }
    }

    /// Defines a query to provide this system for its execution. Multiple queries can be provided,
    /// and queries are cached internally for efficiency for filtering and archetype ID handling.
    ///
    /// It is best practice to define your queries here, to allow for the caching to take place.
    /// These queries are then provided to the executing closure as a tuple of queries.
    pub fn with_query<V, F>(
        mut self,
        query: Query<V, F>,
    ) -> SystemBuilder<<Q as ConsAppend<Query<V, F>>>::Output, R, E>
    where
        V: IntoView,
        F: 'static + EntityFilter,
        Q: ConsAppend<Query<V, F>>,
    {
        self.component_access.add(V::View::requires_permissions());

        SystemBuilder {
            name: self.name,
            queries: ConsAppend::append(self.queries, query),
            events: self.events,
            event_types: self.event_types,
            resources: self.resources,
            resource_access: self.resource_access,
            component_access: self.component_access,
            access_all_archetypes: self.access_all_archetypes,
        }
    }

    /// Flag this resource type as being read by this system.
    ///
    /// This will inform the dispatcher to not allow any writes access to this resource while
    /// this system is running. Parralel reads still occur during execution.
    pub fn read_resource<T>(mut self) -> SystemBuilder<Q, <R as ConsAppend<Read<T>>>::Output, E>
    where
        T: 'static + Resource,
        R: ConsAppend<Read<T>>,
        <R as ConsAppend<Read<T>>>::Output: ConsFlatten,
    {
        self.resource_access.push_read(ResourceTypeId::of::<T>());

        SystemBuilder {
            name: self.name,
            queries: self.queries,
            events: self.events,
            event_types: self.event_types,
            resources: ConsAppend::append(self.resources, Read::<T>::default()),
            resource_access: self.resource_access,
            component_access: self.component_access,
            access_all_archetypes: self.access_all_archetypes,
        }
    }

    /// Flag this resource type as being written by this system.
    ///
    /// This will inform the dispatcher to not allow any parallel access to this resource while
    /// this system is running.
    pub fn write_resource<T>(mut self) -> SystemBuilder<Q, <R as ConsAppend<Write<T>>>::Output, E>
    where
        T: 'static + Resource,
        R: ConsAppend<Write<T>>,
        <R as ConsAppend<Write<T>>>::Output: ConsFlatten,
    {
        self.resource_access.push(ResourceTypeId::of::<T>());

        SystemBuilder {
            name: self.name,
            queries: self.queries,
            events: self.events,
            event_types: self.event_types,
            resources: ConsAppend::append(self.resources, Write::<T>::default()),
            resource_access: self.resource_access,
            component_access: self.component_access,
            access_all_archetypes: self.access_all_archetypes,
        }
    }

    /// Flag this event type as being required by this system.
    pub fn request_event<T>(mut self) -> SystemBuilder<Q, R, <E as ConsAppend<Read<T>>>::Output>
    where
        T: 'static + Event,
        E: ConsAppend<Read<T>>,
        <E as ConsAppend<Read<T>>>::Output: ConsFlatten,
    {
        self.event_types.push(EventTypeId::of::<T>());

        SystemBuilder {
            name: self.name,
            queries: self.queries,
            resources: self.resources,
            resource_access: self.resource_access,
            events: ConsAppend::append(self.events, Read::<T>::default()),
            event_types: self.event_types,
            component_access: self.component_access,
            access_all_archetypes: self.access_all_archetypes,
        }
    }

    /// This performs a soft resource block on the component for writing. The dispatcher will
    /// generally handle dispatching read and writes on components based on archetype, allowing
    /// for more granular access and more parallelization of systems.
    ///
    /// Using this method will mark the entire component as read by this system, blocking writing
    /// systems from accessing any archetypes which contain this component for the duration of its
    /// execution.
    ///
    /// This type of access with `SubWorld` is provided for cases where sparse component access
    /// is required and searching entire query spaces for entities is inefficient.
    pub fn read_component<T>(mut self) -> Self
    where
        T: Component,
    {
        self.component_access.push_read(ComponentTypeId::of::<T>());
        self.access_all_archetypes = true;

        self
    }

    /// This performs a exclusive resource block on the component for writing. The dispatcher will
    /// generally handle dispatching read and writes on components based on archetype, allowing
    /// for more granular access and more parallelization of systems.
    ///
    /// Using this method will mark the entire component as written by this system, blocking other
    /// systems from accessing any archetypes which contain this component for the duration of its
    /// execution.
    ///
    /// This type of access with `SubWorld` is provided for cases where sparse component access
    /// is required and searching entire query spaces for entities is inefficient.
    pub fn write_component<T>(mut self) -> Self
    where
        T: Component,
    {
        self.component_access.push(ComponentTypeId::of::<T>());
        self.access_all_archetypes = true;

        self
    }

    /// Builds a system which is not `Schedulable`, as it is not thread safe (!Send and !Sync),
    /// but still implements all the calling infrastructure of the `Runnable` trait. This provides
    /// a way for legion consumers to leverage the `System` construction and type-handling of
    /// this build for thread local systems which cannot leave the main initializing thread.
    pub fn build<F>(
        self,
        run_fn: F,
    ) -> System<<E as ConsFlatten>::Output, <R as ConsFlatten>::Output, <Q as ConsFlatten>::Output, F>
    where
        <E as ConsFlatten>::Output: for<'a> EventSet<'a>,
        <R as ConsFlatten>::Output: for<'a> ResourceSet<'a>,
        <Q as ConsFlatten>::Output: QuerySet,
        F: FnMut(
            &mut CommandBuffer,
            &mut SubWorld,
            &<<E as ConsFlatten>::Output as EventSet<'static>>::Result,
            &mut <<R as ConsFlatten>::Output as ResourceSet<'static>>::Result,
            &mut <Q as ConsFlatten>::Output,
        ) -> SystemResult,
    {
        System {
            name: self.name,
            run_fn,
            _events: EventMarker(PhantomData),
            _resources: ResourceMarker(PhantomData),
            event_types: self.event_types,
            queries: self.queries.flatten(),
            archetypes: if self.access_all_archetypes {
                ArchetypeAccess::All
            } else {
                ArchetypeAccess::Some(BitSet::default())
            },
            access: SystemAccess {
                resources: self.resource_access,
                components: self.component_access,
            },
            command_buffer: HashMap::default(),
        }
    }
}

