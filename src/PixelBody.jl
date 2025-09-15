using WaterLily,StaticArrays
import WaterLily: AbstractBody, measure, measure!, measure_sdf!

"""
    PixelBody{T,A<:AbstractArray{T,2}} <: AbstractBody

A body derived from a pixel image.
"""
struct PixelBody{T,A<:AbstractArray{T,2}} <: AbstractBody
    R::CartesianIndices # subarray range
    mask::A; buff::A # integrated mask and ping-pong buffer
    function PixelBody(mask::AbstractArray{T,2},dims;mem=Array) where T
        n,m = dims .+ 2
        W,H = size(mask)
        W₀,H₀ = min(W,ceil(Int,n/m*H)),min(H,ceil(Int,m/n*W))
        R = CartesianIndices(((W-W₀)÷2+1:(W+W₀)÷2,(H-H₀)÷2+1:(H+H₀)÷2))
        buff = mem{T}(undef,size(R)...)
        a = new{T,typeof(buff)}(R,similar(buff),buff) # uninitialized
        update!(a,mask); a # initialize
    end
end
function update!(a::PixelBody{T},mask::AbstractArray{T,2}) where T
    copyto!(a.mask,CartesianIndices(a.mask),mask,a.R)
    cumsum!(a.buff,a.mask,dims=1)
    cumsum!(a.mask,a.buff,dims=2) # ping-pong
end
"""
    measure!(a::Flow,b::PixelBody)

Measures zeroth-moment μ₀ by integrating the image mask
"""
function measure!(a::Flow{2,T},b::PixelBody;ϵ=1,kwargs...) where T
    a.V .= zero(T); a.σ .= one(T); a.μ₀ .= one(T); a.μ₁ .= zero(T) # init

    # zeroth-moment is the volume-fraction of masked pixels within a cell
    W,H = size(b.mask); n,m = size(a.σ)
    function vol_frac(ij,mask)
        i,j = ij.I
        I₀,J₀ = clamp(floor(Int,(i-ϵ)*W/n),1,W),clamp(floor(Int,(j-ϵ)*H/m),1,H)
        I₁,J₁ = clamp(ceil(Int,(i+ϵ)*W/n),1,W),clamp(ceil(Int,(j+ϵ)*H/m),1,H)
        dv = T((I₁-I₀)*(J₁-J₀))
        (mask[I₀,J₀]+mask[I₁,J₁]-mask[I₀,J₁]-mask[I₁,J₀])/dv
    end
    WaterLily.@loop a.σ[I] = vol_frac(I,b.mask) over I ∈ inside(a.σ,buff=ceil(Int,ϵ))

    # Interpolate to faces
    for i ∈ 1:2
        WaterLily.@loop a.μ₀[I,i] = WaterLily.ϕ(i,I,a.σ) over I ∈ inside(a.σ)
    end
    WaterLily.BC!(a.μ₀,zeros(SVector{2,T}),false,a.perdir) # BC on μ₀, don't fill normal component yet
end
measure(::PixelBody,x::AbstractVector,args...;kwargs...)=(Inf,zero(x),zero(x)) # can't do this
measure_sdf!(a::AbstractArray,body::PixelBody,t=0;kwargs...) = @warn "Can't do this yet"