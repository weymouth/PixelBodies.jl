using VideoIO,Images,Plots
using JLD2
grayteapot = try
    load_object("grayteapot.jld2")
catch
    cam = VideoIO.opencamera("video=Integrated Webcam")
    color = VideoIO.read(cam)
    VideoIO.close(cam)
    grayteapot = Gray.(color)
    save_object("grayteapot.jld2",grayteapot)
end
mask = reverse(grayteapot',dims=2) .> 0.25

using WaterLily,PixelBodies
dims = (256,256)
sim = Simulation(dims,(1,0),128,body = PixelBody(mask,dims));
WaterLily.flood(sim.flow.σ)
sim_step!(sim)
@assert sim.pois.n[end]<3
sim_gif!(sim,duration=3,clims=(-50,50))
