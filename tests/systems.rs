use legion::*;
use world::SubWorld;
use smallvec::SmallVec;
use legion::systems::QuerySet;

/*#[test]
#[cfg(feature = "codegen")]
fn basic_system() {
    struct A;
    #[system]
    fn nou(#[state] st: &u32) -> SystemResult {
        println!("hello world");
        Ok(SmallVec::new())
    }
    #[system]
    fn hello_world(#[query] _query1: <&A>::query()) -> SystemResult {
        println!("hello world");
        Ok(SmallVec::new())
    }

    let mut world = World::default();
    let mut schedule = Schedule::builder().add_system(hello_world_system()).build();

    schedule.execute(&mut world, &Events::default(), &mut Resources::default());
}*/


#[test]
#[cfg(feature = "codegen")]
fn query_get() {
    type State = Entity;
    struct A;

    #[system]
    #[read_component(f32)]
    #[read_component(f64)]
    #[query(<&A>::query())]
    #[query(<&A>::query())]
    fn sys<Queries: QuerySet>(world: &mut SubWorld, #[state] entity: &State, query1: &Queries) -> SystemResult {
        let mut query = <(&f32, &f64)>::query();
        query.get_mut(world, *entity);
        return Ok(SmallVec::new());
    }

    let mut world = World::default();
    let entity = world.push(());

    let mut schedule = Schedule::builder().add_system(sys_system(entity)).build();

    let mut resources = Resources::default();
    let mut events = Events::default();

    schedule.execute(&mut world, &events, &mut resources);
}
