In this project i will explore the capabilities of reinforcement learning. 

A 5 year old child is capable ob beating games like Mario Bros and Pokemon by trial and error even when they dont know how to read so, how does this child know if he is doing okay?
I made this thought and i think the answer is that it pleases their curiosity. The game keeps showing new stuff, levels and designs if they advance, and brings them back to the beggining if they fail.
In fact, this can be aplied in games played at older ages. A game that is repetitive will be boring and you will stop playing.
So how do we model curiosity? There appears to be several aproaches but the one that seems natural to me is to measure the "originality" of what the model is seeing, the same way a person would do. 
The specific implementation might change from experiment to experiment but it will be based in this idea.

Now, some games are harder to play than others. For example Pokemon requires backtracking, which seems hard to learn, so for now i'll stick to Super Mario Land.
Clasic Game Boy games are the perfect challengers as they are grey-scaled and low resolution, which will improve training time.


i have a fairly modest pc (ryzen 5 1600, nvidia gtx 1070) and some test are not feasible in my machine (IE. big models or keeping the trainning running for too much time. If i don't see improvements in a long time i will shut it down.)

The pokemon ROM yet to be used and the discretizer can be found in the following repository:
https://github.com/JFlaherty347/Pokemon-Red-AI