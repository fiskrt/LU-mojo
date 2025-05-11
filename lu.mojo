from gpu import thread_idx, block_dim, block_idx
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from testing import assert_equal
from random import rand

alias N = 6
alias dtype = DType.float32
alias layout = Layout.row_major(N, N)

fn process_pivot(
    a: LayoutTensor[mut=True, dtype, layout],
    pivot_idx: Int
):
    col = thread_idx.x
    
    if col < N:
        if col == pivot_idx:
            let pivot_value = a[pivot_idx, pivot_idx]
            a[pivot_idx, pivot_idx] = 1.0
        elif col > pivot_idx:
            let pivot_value = a[pivot_idx, pivot_idx]
            a[pivot_idx, col] = a[pivot_idx, col] / pivot_value

fn process_rows(
    a: LayoutTensor[mut=True, dtype, layout],
    pivot_idx: Int
):
    row = thread_idx.y
    col = thread_idx.x
    
    if row < N and col < N and row > pivot_idx:
        let factor = a[row, pivot_idx]
        
        if col == pivot_idx:
            a[row, pivot_idx] = 0.0
        elif col > pivot_idx:
            a[row, col] = a[row, col] - (factor * a[pivot_idx, col])

fn main():
    with DeviceContext() as ctx:
        a_buf = ctx.enqueue_create_buffer[dtype](N * N)
        
        with a_buf.map_to_host() as a_host:
            for i in range(N * N):
                a_host[i] = Float32(1.0 + rand(0, 10)[0])
        
        a_tensor = LayoutTensor[mut=True, dtype, layout](a_buf.unsafe_ptr())
        
        print("Original Matrix A:")
        print(a_tensor)
        
        result_buf = ctx.enqueue_create_buffer[dtype](N * N)
        ctx.enqueue_copy_buffer(a_buf, result_buf)
        result_tensor = LayoutTensor[mut=True, dtype, layout](result_buf.unsafe_ptr())
        
        alias THREADS_PER_BLOCK = (N, N)
        alias BLOCKS_PER_GRID = 1
        
        print("\n=== Performing LU Decomposition ===")
        for i in range(N):
            print("Processing pivot", i)
            
            ctx.enqueue_function[process_pivot](
                result_tensor,
                i,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK
            )
            
            ctx.synchronize()
            
            ctx.enqueue_function[process_rows](
                result_tensor,
                i,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK
            )
            
            ctx.synchronize()
        
        print("\nResulting LU Matrix:")
        print(result_tensor)
        
        print("\n=== Verifying LU Decomposition ===")
        
        recon_buf = ctx.enqueue_create_buffer[dtype](N * N)
        recon_tensor = LayoutTensor[mut=True, dtype, layout](recon_buf.unsafe_ptr())
        
        with result_buf.map_to_host() as result_host, recon_buf.map_to_host() as recon_host:
            for i in range(N):
                for j in range(N):
                    var sum: Float32 = 0.0
                    for k in range(N):
                        var l_val: Float32
                        if i > k:
                            l_val = result_host[i * N + k]
                        elif i == k:
                            l_val = 1.0
                        else:
                            l_val = 0.0
                        
                        var u_val: Float32
                        if k <= j:
                            u_val = result_host[k * N + j]
                        else:
                            u_val = 0.0
                        
                        sum += l_val * u_val
                    
                    recon_host[i * N + j] = sum
            
        print("\nReconstructed Matrix A = L*U:")
        print(recon_tensor)
