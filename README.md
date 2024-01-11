# bioacoustics_python

by Léa Bouffaut, Ph.D. -- K. Lisa Yang Center for Conservation Bioacoustics -- lea.bouffaut@cornell.edu

This repository aims to provide helpful code examples for bioacoustics in Python and uses Jupyter Notebooks. At the moment, the following examples are available:
* _AudioManipulating/_, a quick tutorial on how to safely resample audio files and choose the output audio file format
* _ModelPerformancEvaluation/_ is used to evaluate automatic detection algorithms that assign a classification score to audio files that are divided into potentially overlapping audio clips of flexible duration (e.g., BirdNET or Koogu-trained models). It generates precision-recall curves based on Raven selection tables for the ground truth and model output.
* _Spectrogram/_ is a quick tutorial to load an audio file and generate and save your own spectrogram!
* _RavenSelectionTable/_, provides an example of how to load and manipulate Raven selection tables, including a quick how-to to generate your own audio clips!

--- New to Python? ---

Python is a  programming language that can be programmed from many different interfaces. But, using the terminal is not super interactive and can be scary to start (also, for projects, it is impractical).  So, people usually use development environments, e.g., Pycharm or Spyder, to code. It gives you a nicer interface for your projects and usually comes with some help (think, when you write a Word document and have grammar suggestions, auto-completion, etc.). I would recommend starting with Jupyter Notebooks. It is a super friendly environment and allows you to have a lot of interactions with the code.

So first, I'd recommend familiarizing yourself with Python basics and Jupyter Notebooks! Here are a couple of helpful tutorials:
* Python for Beginners - Learn Python in 1 Hour: https://www.youtube.com/watch?v=kqtD5dpn9C8
* Jupyter Lab is AWESOME For Data Science: https://www.youtube.com/watch?v=7wf1HhYQiDg"


_S'more tips from my personal experience_

**Tip 1: Get a grasp of the vocab' and grammar!**
The above videos should help but get familiar with 
* variables (https://realpython.com/python-variables/),
* data types (https://realpython.com/python-data-types/),
* data structures (especially lists, arrays, and dictionaries: https://realpython.com/python-data-structures/), and
* loops (if, for, while: https://www.learnpython.org/en/Loops, https://www.w3schools.com/python/python_for_loops.asp)

**Tip 2: Start with a project that you find fun!**
Some ideas with examples: 
https://www.dataquest.io/blog/python-projects-for-beginners/ 
If your goal is to learn to manipulate data, check out "Fun Python Project Ideas for Building Data Skills" section. If you find it fun, you will want to get back to it, and it'll be easier to dedicate time to figure things out.

**Tip 3: Practice Algorithmic!**
Meaning, structure your code as you would for a research paper. What is the objective of this piece of code? How would you break it down into simpler "logical" tasks? 

**Tip 4: Google your way!**
You do not need to remember all functions. Some, of course, you will remember by practicing. For most, you might have to Google it, and that's okay! What is essential is to learn how to do a Google search to find valuable information. :)

**Tip 5: Document your code!**
It is good practice to document your code, and starting by drafting your English version of the algorithm should help. But even more, don't hesitate to add comments, explain why you're doing things one way and not another etc.

**Tip 6: Use explicit variable names!** 
I remember when I started, all my variables were named x1, x2, y3... It was an absolute hell for other people to read my code. Now, if x1  day and y3 daily_temperature, all of the sudden, it is a lot clearer to everyone! You can use the Tab key for autocompletion. 

**Tip 7: Read other people's code!**
Like learning a foreign language, it will help discover new ways of getting a specific task done and it will help broaden your "code literacy." As for a foreign language you're reading, there might be words (e.g., functions) or grammatical tools (e.g., code structure) you are unfamiliar with: back to tip 4... Google it!
