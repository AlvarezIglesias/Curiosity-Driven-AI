This first test is a quick proof of concept, using a modified Deep Mind's implementation of their atari AI. I have yet to change the comments and structure of the script.

First of all, the implementation of the ROM is not correct and the initial state is not working. Also the game requires to press START to begin, but this button also pauses the game so the model keeps pausing the game. With training, it starts to use it less and less but it is anoying and it would require modifiying the ROM.

In this first experiment, the reward funcions is calculated comparing the new states with all the previous ones, and the reward is the difference between the newest one and the most similar from the past. This value can be 0 if the same exact state was already found.
The state is not saved as a image nor an array, but as a hash to save up space. This arises a lot of problems as two similar images generate two completily different hashes, but this is fixed using ImageHash. 
Even with this in mind, the funcion generates rewards in situations where it shouldn't. IE Mario going to the left causes the screen to stop following him, so we now have a state that doesn't change at all except for Mario in different places. As a result the model learns to go right and then jump at the left of the screen, and then continue right.


The model was trained for 1976000 frames. 
It looks like it was getting better at the end but this was already 12+ hours training.

<img src="/first_test/media/he_tries.gif" width="250" height="250"/>
<img src="/first_test/media/rewards.png" width="250" height="250"/>


