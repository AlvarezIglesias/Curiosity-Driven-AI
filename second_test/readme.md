Changes:

Lower learning rate.
Two layers in the NN instead of one but smaller.
Small penalization for already seen states and reduced rewards when its really similar.
Fixed the savestate and now resets correctly when the episode ends.
Blocked left button to speed up process. (would not do this in the 'final' test)

Results:

The new model definetly plays better than the previous one. In late episodes, it defeated the first goomba most of the times.

Here it defeats a goomba and does a little dance as celebration:

<img src="/media/defeat_and_dance.gif" width="250" height="250"/>

Sadly, it would get stuck in a lot of obstacles. I think the reward function is not correct. When it gets stuck in the first obstacle, the rewards is not significantly lower than when it completes half the level. I will change it for the next try


<img src="/media/rewards2.png" width="300" height="250"/>
