module {
  llvm.func @wrap_elementwise(!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
  llvm.func @hipdnn_ep_alloc_output(!llvm.ptr, i64, !llvm.ptr, i64, i64) -> !llvm.ptr
  llvm.func @main_graph(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: !llvm.ptr, %arg11: !llvm.ptr, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: i64) -> !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> {
    %0 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %1 = llvm.insertvalue %arg10, %0[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2 = llvm.insertvalue %arg11, %1[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %3 = llvm.insertvalue %arg12, %2[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %4 = llvm.insertvalue %arg13, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %5 = llvm.insertvalue %arg16, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %6 = llvm.insertvalue %arg14, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %7 = llvm.insertvalue %arg17, %6[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %8 = llvm.insertvalue %arg15, %7[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %9 = llvm.insertvalue %arg18, %8[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %10 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %11 = llvm.insertvalue %arg1, %10[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %12 = llvm.insertvalue %arg2, %11[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %13 = llvm.insertvalue %arg3, %12[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %14 = llvm.insertvalue %arg4, %13[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %15 = llvm.insertvalue %arg7, %14[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %16 = llvm.insertvalue %arg5, %15[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %17 = llvm.insertvalue %arg8, %16[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %18 = llvm.insertvalue %arg6, %17[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %19 = llvm.insertvalue %arg9, %18[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %20 = llvm.mlir.constant(1 : index) : i64
    %21 = llvm.mlir.constant(4 : index) : i64
    %22 = llvm.mlir.constant(4 : index) : i64
    %23 = llvm.mlir.constant(1 : index) : i64
    %24 = llvm.mlir.constant(16 : index) : i64
    %25 = llvm.mlir.constant(16 : index) : i64
    %26 = llvm.mlir.zero : !llvm.ptr
    %27 = llvm.getelementptr %26[%25] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %28 = llvm.ptrtoint %27 : !llvm.ptr to i64
    %29 = llvm.mlir.constant(1 : i64) : i64
    %30 = llvm.alloca %29 x !llvm.array<3 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %31 = llvm.mlir.constant(0 : i32) : i32
    %32 = llvm.getelementptr %30[%31] : (!llvm.ptr, i32) -> !llvm.ptr, i64
    llvm.store %20, %32 : i64, !llvm.ptr
    %33 = llvm.mlir.constant(1 : i32) : i32
    %34 = llvm.getelementptr %30[%33] : (!llvm.ptr, i32) -> !llvm.ptr, i64
    llvm.store %21, %34 : i64, !llvm.ptr
    %35 = llvm.mlir.constant(2 : i32) : i32
    %36 = llvm.getelementptr %30[%35] : (!llvm.ptr, i32) -> !llvm.ptr, i64
    llvm.store %22, %36 : i64, !llvm.ptr
    %37 = llvm.mlir.constant(0 : i64) : i64
    %38 = llvm.mlir.constant(3 : i64) : i64
    %39 = llvm.mlir.constant(4 : i64) : i64
    %40 = llvm.call @hipdnn_ep_alloc_output(%arg0, %37, %30, %38, %39) : (!llvm.ptr, i64, !llvm.ptr, i64, i64) -> !llvm.ptr
    %41 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %42 = llvm.insertvalue %40, %41[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %43 = llvm.insertvalue %40, %42[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %44 = llvm.mlir.constant(0 : index) : i64
    %45 = llvm.insertvalue %44, %43[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %46 = llvm.insertvalue %20, %45[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %47 = llvm.insertvalue %21, %46[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %48 = llvm.insertvalue %22, %47[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %49 = llvm.insertvalue %24, %48[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %50 = llvm.insertvalue %22, %49[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %51 = llvm.insertvalue %23, %50[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %52 = llvm.mlir.constant(1 : i64) : i64
    %53 = llvm.mlir.constant(1 : i64) : i64
    %54 = llvm.mlir.constant(4 : i64) : i64
    %55 = llvm.mlir.constant(4 : i64) : i64
    %56 = llvm.mlir.constant(1 : i64) : i64
    %57 = llvm.mlir.constant(1 : i64) : i64
    %58 = llvm.mlir.constant(4 : i64) : i64
    %59 = llvm.mlir.constant(4 : i64) : i64
    %60 = llvm.mlir.constant(1 : i64) : i64
    %61 = llvm.mlir.constant(1 : i64) : i64
    %62 = llvm.mlir.constant(4 : i64) : i64
    %63 = llvm.mlir.constant(4 : i64) : i64
    %64 = llvm.mlir.constant(-1 : i32) : i32
    %65 = llvm.extractvalue %19[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %66 = llvm.extractvalue %9[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %67 = llvm.extractvalue %51[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %68 = llvm.mlir.constant(0 : i64) : i64
    %69 = llvm.mlir.constant(1 : i64) : i64
    %70 = llvm.call @wrap_elementwise(%arg0, %64, %65, %66, %67, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %68, %69) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    llvm.return %51 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
  }
}

