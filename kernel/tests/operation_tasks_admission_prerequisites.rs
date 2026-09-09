#![cfg(feature = "operation-tasks")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;

use delta_kernel::actions::deletion_vector::DeletionVectorDescriptor;
use delta_kernel::expressions::ColumnName;
use delta_kernel::plans::ir::nodes::{DynamicScan, FileType};
use delta_kernel::schema::{DataType, StructField, StructType, ToSchema};

thread_local! {
    static ALLOCATIONS: Cell<Option<usize>> = const { Cell::new(None) };
}

struct CountingAllocator;

fn count_allocation() {
    let _ = ALLOCATIONS.try_with(|count| {
        if let Some(n) = count.get() {
            count.set(Some(n + 1));
        }
    });
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count_allocation();
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count_allocation();
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count_allocation();
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn dynamic_scan_revalidation_does_not_allocate_column_path_vectors() {
    for depth in [0, 3, 32] {
        let mut schema = StructType::try_new([
            StructField::not_null("path", DataType::STRING),
            StructField::not_null("size", DataType::LONG),
            StructField::not_null("modified", DataType::LONG),
            StructField::nullable("dv", DeletionVectorDescriptor::to_schema()),
        ])
        .unwrap();
        for _ in 0..depth {
            schema = StructType::try_new([StructField::not_null("nested", schema)]).unwrap();
        }
        let input = Arc::new(schema);
        let column = |leaf: &str| {
            ColumnName::new(std::iter::repeat_n("nested", depth).chain(std::iter::once(leaf)))
        };
        // Construction initializes the shared DV schema outside the measured revalidation.
        let scan = DynamicScan::try_new(
            &input,
            Arc::new(StructType::try_new([]).unwrap()),
            FileType::Parquet,
            "memory:///".parse().unwrap(),
            std::iter::empty::<String>(),
            column("path"),
            column("size"),
            column("modified"),
            column("dv"),
        )
        .unwrap();
        ALLOCATIONS.with(|count| count.set(Some(0)));
        let result = scan.validate_input(&input);
        let allocations = ALLOCATIONS.with(|count| count.replace(None).unwrap());
        result.unwrap();
        assert_eq!(allocations, 0, "path depth {depth}");
    }
}
