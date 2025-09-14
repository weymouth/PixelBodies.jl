using WaterLily,StaticArrays
import WaterLily: AbstractBody, measure, measure!, measure_sdf!

"""
    PixelBody{T,A<:AbstractArray{T,2}} <: AbstractBody

A body derived from a pixel image.
"""
struct PixelBody{A<:AbstractArray{Int,2}} <: AbstractBody
    R::CartesianIndices # subarray range
    mask::A; hold::A # integrated mask
    function PixelBody(mask::AbstractArray{Bool,2},dims;mem=Array)
        n,m = dims .+ 2
        W,H = size(mask)
        W₀,H₀ = min(W,ceil(Int,n/m*H)),min(H,ceil(Int,m/n*W))
        R = CartesianIndices(((W-W₀)÷2+1:(W+W₀)÷2,(H-H₀)÷2+1:(H+H₀)÷2))
        hold = copy(mask[R]) |> mem .|> Int #copy subarray 
        mask = cumsum(cumsum(hold,dims=1),dims=2)
        new{typeof(mask)}(R,mask,hold)
    end
end
function update!(a::PixelBody,mask::AbstractArray{Bool,2})
    copyto!(a.mask,view(mask[a.R]))
    cumsum!(a.hold,a.mask,dims=1)
    cumsum!(a.mask,a.hold,dims=2) # ping-pong
end
"""
    measure!(a::Flow,b::PixelBody)

Measures zeroth-moment μ₀ by integrating the image mask
"""
function measure!(a::Flow{2,T},b::PixelBody;kwargs...) where T
    a.V .= zero(T); a.σ .= one(T); a.μ₀ .= one(T); a.μ₁ .= zero(T) # init

    # μ₀ is the volume-fraction of masked pixels within a cell
    W,H = size(b.mask); n,m = size(a.σ)
    for i in 2:n-1, j in 2:m-1
        I₀,J₀ = clamp(floor(Int,(i-1)*W/n),1,W),clamp(floor(Int,(j-1)*H/m),1,H)
        I₁,J₁ = clamp(ceil(Int,(i+1)*W/n),1,W),clamp(ceil(Int,(j+1)*H/m),1,H)
        dv = T((I₁-I₀)*(J₁-J₀))
        a.σ[i,j] = (b.mask[I₀,J₀]+b.mask[I₁,J₁]-b.mask[I₀,J₁]-b.mask[I₁,J₀])/dv
    end

    # Interpolate to faces
    for i ∈ 1:2
        WaterLily.@loop a.μ₀[I,i] = WaterLily.ϕ(i,I,a.σ) over I ∈ inside(a.σ)
    end
    WaterLily.BC!(a.μ₀,zeros(SVector{2,T}),false,a.perdir) # BC on μ₀, don't fill normal component yet
end
measure(::PixelBody,x::AbstractVector,args...;kwargs...)=(Inf,zero(x),zero(x)) # can't do this
measure_sdf!(a::AbstractArray,body::PixelBody,t=0;kwargs...) = @warn "Can't do this yet"