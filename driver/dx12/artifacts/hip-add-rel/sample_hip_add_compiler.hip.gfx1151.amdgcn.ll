; ModuleID = 'C:\Users\sambaner\workspace3\hip-ep\driver\dx12\artifacts\hip-add-rel\sample_hip_add_compiler.hip.dx12-device.hip'
source_filename = "C:\\Users\\sambaner\\workspace3\\hip-ep\\driver\\dx12\\artifacts\\hip-add-rel\\sample_hip_add_compiler.hip.dx12-device.hip"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

@__hip_cuid_f974d40b670b27a4 = addrspace(1) global i8 0
@llvm.compiler.used = appending addrspace(1) global [1 x ptr] [ptr addrspacecast (ptr addrspace(1) @__hip_cuid_f974d40b670b27a4 to ptr)], section "llvm.metadata"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define protected amdgpu_kernel void @hip_ep_add_f32(ptr addrspace(1) noalias noundef readonly captures(none) %0, ptr addrspace(1) noalias noundef readonly captures(none) %1, ptr addrspace(1) noalias noundef writeonly captures(none) %2, i64 noundef %3) local_unnamed_addr #0 {
  %5 = tail call i32 @llvm.amdgcn.workgroup.id.x()
  %6 = zext i32 %5 to i64
  %7 = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %8 = getelementptr inbounds nuw i8, ptr addrspace(4) %7, i64 12
  %9 = load i16, ptr addrspace(4) %8, align 4, !tbaa !7
  %10 = zext i16 %9 to i64
  %11 = mul nuw nsw i64 %10, %6
  %12 = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x()
  %13 = zext nneg i32 %12 to i64
  %14 = add nuw nsw i64 %11, %13
  %15 = icmp slt i64 %14, %3
  br i1 %15, label %16, label %23

16:                                               ; preds = %4
  %17 = getelementptr inbounds nuw float, ptr addrspace(1) %2, i64 %14
  %18 = getelementptr inbounds nuw float, ptr addrspace(1) %1, i64 %14
  %19 = getelementptr inbounds nuw float, ptr addrspace(1) %0, i64 %14
  %20 = load float, ptr addrspace(1) %19, align 4
  %21 = load float, ptr addrspace(1) %18, align 4
  %22 = fadd contract float %20, %21
  store float %22, ptr addrspace(1) %17, align 4
  br label %23

23:                                               ; preds = %16, %4
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef align 4 ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr() #1

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "amdgpu-agpr-alloc"="0" "amdgpu-flat-work-group-size"="1,1024" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1151" "target-features"="+16-bit-insts,+atomic-fadd-rtn-insts,+ci-insts,+dl-insts,+dot10-insts,+dot12-insts,+dot5-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx8-insts,+gfx9-insts,+wavefrontsize32" "uniform-work-group-size"="true" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!opencl.ocl.version = !{!5}
!llvm.ident = !{!6}

!0 = !{i32 1, !"amdhsa_code_object_version", i32 600}
!1 = !{i32 1, !"amdgpu_printf_kind", !"hostcall"}
!2 = !{i32 1, !"wchar_size", i32 2}
!3 = !{i32 8, !"PIC Level", i32 2}
!4 = !{i32 1, !"MaxTLSAlign", i32 65536}
!5 = !{i32 2, i32 0}
!6 = !{!"clang version 21.0.0git (git@github.com:AMD-Lightning-Internal/llvm-project 590b9320a5be90e40268759c6203c01fde121e68)"}
!7 = !{!8, !8, i64 0}
!8 = !{!"short", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
