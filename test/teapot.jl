using JLD2,Plots
grayteapot = try
    load_object("grayteapot.jld2")
catch
    using VideoIO,Images
    cam = VideoIO.opencamera("video=Integrated Webcam")
    color = VideoIO.read(cam)
    VideoIO.close(cam)
    grayteapot = Gray.(color)
    save_object("grayteapot.jld2",grayteapot)
end
mask = (reverse(grayteapot',dims=2) .> 0.25) .|> UInt32 # binary mask

using WaterLily,PixelBodies,CUDA
dims = (256,256); mem = Array
sim = Simulation(dims,(1,0),128; body=PixelBody(mask,dims; mem), T=Float32, mem);
WaterLily.flood(sim.flow.σ,c=:greys)
sim_step!(sim)
@time PixelBodies.update!(sim.body,mask); # if mask changes
@assert sim.pois.n[end]<3
sim_gif!(sim,duration=3,clims=(-50,50),remeasure=false)
