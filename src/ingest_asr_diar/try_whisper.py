from faster_whisper import WhisperModel
from pathlib import Path
import time

start = time.time()
model_size = "models/faster-whisper-medium"
model_path = Path(model_size)
model = WhisperModel(str(model_path), device="cpu", compute_type="int8")
segments, info = model.transcribe("data/raw/S16.wav", beam_size=5)
print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
end = time.time()

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

print(f"Transcription completed in {end - start:.2f} seconds")
# Transcription completed in 12.70 seconds

"""
Results:
Detected language 'en' with probability 0.996885
[0.00s -> 11.80s]  start with categorization. So what types of trash will be expected to find and we
[11.80s -> 18.16s]  would want to categorize them. Basically nothing will be reusable as it is. An
[18.16s -> 24.90s]  ocean beach trash. Most of it will just be different types of material almost all
[24.90s -> 32.46s]  right so then like you said a plastic the glass and metal that's well yeah
[32.46s -> 39.26s]  plastic metal so environmentally nothing is really good to be recycled but there's
[39.26s -> 45.18s]  use cases so like glass is one of the only like almost entirely recyclable
[45.18s -> 52.26s]  things but they put shit in the glass too so that's no point
[52.46s -> 61.02s]  let's okay generally for for plastics you can melt them all down technically and
[61.02s -> 68.30s]  like and then introduce them into some kind of like newer plastic for like 3d
[68.30s -> 75.50s]  printing yeah again like it's not very good you still have to add it to new
[76.10s -> 84.22s]  and then every every old recycled plastic is like a poor quality finished product
[84.22s -> 90.74s]  so it's a possibility to use recycled plastics as schools but not on a
[90.74s -> 100.34s]  practical level personally I'm like I've got a lot of opinions about recycling
[100.38s -> 106.86s]  we should everything that's not naturally decomposable should be really
[107.22s -> 112.22s]  yeah well I've seen do effective upcycling or like certain artists
[112.22s -> 119.90s]  last can be used as a replacement for paying windows if you've ever seen or
[119.90s -> 125.78s]  heard of earthships they do like a radical sustainable housing construction
[125.78s -> 128.66s]  projects and like the middle of the desert the only place they're allowed
[128.66s -> 134.62s]  to try anything housing right I've never seen that but I've like I've seen I've
[134.62s -> 142.98s]  met artists like so most of them they try to do like local natural materials
[142.98s -> 148.42s]  and recycled products so one thing is they'll use tires and there's not a
[148.42s -> 151.66s]  whole lot of things you can do with tires in fact they they often just
[151.66s -> 156.22s]  burn them so there's like all this crap rubber and plastic is just
[156.42s -> 166.18s]  into the atmosphere so the tires are the basis of the walls and then like a
[166.18s -> 174.50s]  window pane almost a little bit further than what I would do but I like there
[174.50s -> 179.82s]  I like their initiative I wonder if the process like you know burning tires
[179.82s -> 186.86s]  creates more you know like carbon dioxide and harmful oh it all it all
[186.86s -> 191.18s]  contributes carbon dioxide and other pollutants too so like you don't want
[191.18s -> 195.58s]  to burn shit anything that's not like made from nature you don't want to
[195.58s -> 204.10s]  burn it really but in our human economy we do nothing in a like
[204.10s -> 215.26s]  practical way or long-term sustainable way certain things like fiber paper
[215.26s -> 220.66s]  cardboard that stuff can actually be decomposed so you can put it in like
[220.66s -> 228.70s]  composting you know and it can be broken down plastics is one of those
[228.70s -> 233.50s]  things that like it's it's a human product that's never really been
[233.50s -> 242.54s]  naturally occurring so there's nothing on earth that really decomposes it I
[242.54s -> 249.34s]  mean one of the reasons we use it but they don't consider long-term thinking
[249.34s -> 256.50s]  and anything in our economy so some scientists try to do like fungal
[256.50s -> 266.38s]  decompositions like GMO fungus but all this is just using energy to get rid of
[266.38s -> 278.98s]  stuff yeah plastic doesn't have a lot of upcycling things like old clothes
[279.22s -> 290.02s]  can be used for just for being soft some animals can repurpose it and like the
[290.02s -> 295.94s]  scene like birds collect cigarette and put them in their nests it actually
[295.94s -> 301.66s]  acts as like an insulation that helps them keep them warm which all those
[301.66s -> 306.98s]  like birds are probably getting sick yeah
[309.42s -> 327.50s]  oh I think we just we need better recycling technology overall so we just
[327.50s -> 332.26s]  so like plastics and all the forever chemicals like we just need to recycle
[332.26s -> 346.30s]  somehow but we don't really have the technology to do well currently some
[346.30s -> 352.30s]  artists like using it but it's pretty low-impact low-effort it's only so
[352.30s -> 357.86s]  much they can do and they're not really still taking up space yeah
[358.78s -> 365.94s]  technically take up even more space yeah and not contributing to them they're
[365.94s -> 376.78s]  still there there's just so much trash it's even like altering the ecology we
[376.78s -> 382.86s]  seem like there's hermit crabs that'll take like soda cans and shit pretend
[382.86s -> 397.34s]  like they're for actual shows like glass trash is it does that so like there
[397.34s -> 403.34s]  the animals are repurposing it instead of the people just by it just because
[403.34s -> 413.90s]  there's just so much trash in the environment but yeah tires can be useful
[413.90s -> 417.86s]  because you could just bury the shit so it it actually provides a good basis
[417.86s -> 423.18s]  for the walls in these earthship houses though the buildings are
[423.18s -> 428.90s]  essentially limited to only be like one story maybe two if they're really
[428.90s -> 440.98s]  clever and when I grew up there's a lot of tire swings the bar is recycled
[440.98s -> 450.82s]  repurposed tires some of the some tires that were shredded were recycled as
[450.82s -> 457.94s]  like a playground like a playground soft padding except that they tires
[457.98s -> 465.22s]  they're still toxic so like they just got replaced with a look-alike they
[465.22s -> 469.62s]  actually just started making rubber pellets that looked like recycle tires
[469.62s -> 480.14s]  to do the same yeah I leave this off use economic approaches through and
[480.34s -> 506.62s]  so when I get plastic bags at home I do upcycling and I find some other thing
[506.66s -> 512.18s]  that I'm gonna need a plastic bag for and that's what I use it for in my case I
[512.18s -> 518.30s]  use it for the kitty litter because kitty litter is kind of disgusting and you
[518.30s -> 523.42s]  want to be pretty well contained yeah and it'll even like seep through the
[523.42s -> 529.22s]  plastic bags but it gives me a few minutes to to get it in the right
[529.22s -> 537.46s]  place sleeping through so get it into a larger bag where it will be absorbed
[537.46s -> 545.50s]  another trash and then that trash bag to the dumpster so you can just keep
[545.50s -> 549.98s]  using the same plastic bag over and over again if they're not coming
[549.98s -> 557.70s]  contaminate why I don't reuse I only upcycle each one once shopping
[557.82s -> 576.10s]  plastic bags and a drawer and then just when next when I needed bags in general
[576.10s -> 581.06s]  just like trash just adds up there's no effective way to keep reusing it but
[581.06s -> 588.02s]  we keep producing more trash the only thing that it would really the things
[588.02s -> 593.94s]  that decompose or compost are the only things that actually like get recycled
[593.94s -> 598.90s]  pulled into the environment so things like those old egg crates they're
[598.90s -> 603.10s]  basically cardboard even that those themselves are probably the actual
[603.10s -> 611.70s]  recycled product from paper but those are also like low level of fibrous
[611.70s -> 616.62s]  product they put plastic and everything these days though so even our clothes
[616.62s -> 628.42s]  are predominantly plastic it's a depressing topic I thought we were yeah
[628.42s -> 634.90s]  so we're gonna come like creative sorry I can get I can get sidetracked too
[634.90s -> 639.34s]  easily so are we getting sidetracked no no we're on the track we're just
[639.34s -> 646.74s]  talking about depressing how depressing it is like but I was that the tire
[646.74s -> 653.26s]  used is weird we're clever because tires start talking about the
[653.26s -> 657.86s]  microplastic majority of microplastics in the waterways are sourced
[657.94s -> 663.30s]  from tires because we just have so many cars on the roads and it's all exposed
[663.30s -> 667.34s]  to the environment all the time
[669.38s -> 676.26s]  and there's no way of but I was very impressed with the upcycling basically
[676.26s -> 682.34s]  would they call it rammed earth they stick a whole bunch of dirt inside a
[682.34s -> 690.06s]  tire and basically buried the tire in a clay or about a wall so it's
[690.06s -> 695.94s]  completely encapsulated inside of your structure but it gives insulation and
[695.94s -> 700.74s]  stability to your wall so those buildings that have these tires in the
[700.74s -> 706.10s]  walls actually very resilient to earthquakes because like the shifting
[706.10s -> 710.86s]  around it's not a very like it's not a very rigid structure and actually
[710.86s -> 715.58s]  like withstands earthquakes really well it also resists fire really well so
[715.58s -> 720.94s]  it's actually like an ideal building environment for so yeah but they'll
[720.94s -> 726.78s]  never approve it because it's not traditional and probably it's not like
[726.78s -> 730.30s]  doesn't create more profit when you're building a building like if you're
[730.30s -> 734.66s]  recycling then it's last profit and they've ever approved it they'll just
[734.66s -> 741.42s]  like make fake tires do it produce them from a from a factory yeah because
[741.42s -> 749.54s]  of jobs or something yeah like yeah and then what about plastic I mean like
[749.54s -> 755.94s]  plastic is I'm well we have a lot more electronic waste these days too
[755.94s -> 761.22s]  wasn't as much when I was younger but these days like so much electronic
[761.22s -> 767.58s]  waste even different more depressing topic planned obsolescence and all these
[767.58s -> 771.90s]  electronic devices are products then they literally design them so they can't
[771.90s -> 776.90s]  be reused you can't reuse components you can't upgrade components you are
[776.90s -> 781.86s]  incentivized to continue to buy entire product over and over again so
[781.86s -> 787.30s]  we have a lot of electronics trash which is mostly plastic with toxic
[787.30s -> 795.82s]  heavy metals we can't do anything about it there are some people that mine like
[795.82s -> 800.78s]  the very old electronics had some kind of gold plating and so they love
[800.78s -> 810.50s]  finding old electronics mining out the gold and silver in larger amounts
[811.38s -> 817.94s]  could you possibly use plastic for buildings like how are you doing it no
[817.94s -> 824.46s]  plastic like plastic doesn't decompose but it does lose its like structural
[824.46s -> 830.58s]  characteristics it gets weaker and it's just like it's like a low-level
[830.58s -> 838.86s]  toxin that nothing can interact with yet it's just like we just pile it up
[838.86s -> 843.42s]  and like if the recyclers won't do anything about it then there's
[843.42s -> 848.26s]  literally nothing we could do is just pile it up until we decide to deal
[848.26s -> 858.30s]  with it so the same thing with so the same environment comes through
[858.30s -> 863.26s]  with the electric components except that like they're all multiple heavy
[863.26s -> 869.90s]  metal types of different structures you could probably just melt them into a
[869.90s -> 873.66s]  big ball of different heavy metals and search them out later on when there's
[873.66s -> 881.86s]  better technology well I like their building for tires yeah I'm biased of
[882.38s -> 893.98s]  for reasons disclosed we don't yeah I like that one too we don't do we have
[893.98s -> 903.34s]  to spend five minutes we're done fine I'm not thinking of anything else to
[903.34s -> 910.78s]  help the task we could be done just with
"""