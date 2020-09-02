//! Automatic query scheduling and parallel execution.

pub use crate::internals::systems::{
    command::{CommandBuffer, WorldWritable},
    events::{
        Event, EventSet, EventTypeId, Events, SyncEvents, UnsafeEvents,
    },
    resources::{
        Fetch, Resource, ResourceSet, ResourceTypeId, Resources, SyncResources, UnsafeResources,
    },
    schedule::{Builder, Executor, ParallelRunnable, Runnable, Schedule, Step},
    system::{QuerySet, System, SystemAccess, SystemBuilder, SystemFn, SystemId},
};
