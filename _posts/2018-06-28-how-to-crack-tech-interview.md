---
layout: post
title: How to crack the tech interview
subtitle: Seven Step Approach
tags: [algorithm, coding interview, tech interview]
---

Interviews are supposed to be difficult. The norm is that you won’t know how to solve a question as soon as you hear it. You will struggle through it, get a bit of help from the interviewer, and arrive at a better solution than what you started with.

When you get a hard question, don’t panic. Just start talking aloud about how you would solve it.

The following seven-step approach works well for many problems:

1. Understand the question. If there’s anything you didn’t understand, clarify it here. Pay special attention to any specific details provided in the question, such as that the input is sorted. You need all those details.
2. Draw an example. Solving questions in your head is very different; get up to the whiteboard and draw an example. It should be a good example, too. The example should be reasonably large and not a special case.
3. Design a brute force algorithm. If there’s a brute force/naïve approach, or even a solution that only partially works, explain it. It’s a starting point, and ensures that your interviewer knows that you’ve gotten at least that far.
4. Optimize the brute force. Not always, but very often, there’s a path from the brute force to the optimal solution.
5. Understand the code. Once you have an optimal algorithm, take a moment to really understand your algorithm. It’s well worth it to not dive into code yet.
6. Implement the code. If you’re comfortable with your process, go ahead and implement it. Don’t be afraid to go back to your example, though, if you start to get confused.
7. Flawless whiteboard coding is rare. If you find mistakes, that’s okay. Analyze why you made the mistake and try to fix it.
 
And, remember: you’re not done until the interviewer says that you’re done! This goes for both the algorithm part and the code part. When you come up with an algorithm, start thinking about the problems accompanying it. When you write code, start trying to find bugs. The vast majority of candidates make mistakes.


### Step 1: Understand the Question

Technical problems are more ambiguous than they might appear, so make sure to ask questions to resolve anything that might be unclear. You may eventually wind up with a very different—or much easier—problem than you had initially thought. In fact, some interviewers will especially test to see if you ask good questions. A question like “Design an algorithm to sort a list” might turn into “Sort a sequence of values between 1 and 10 that are stored in a linked list.” This is a very different problem.

Additionally, it’s important to make sure that you really remember all those details that the interviewer mentioned. If the interviewer mentioned that the data is stored, then your optimal algorithm probably depends on that. Or, if the data set has all unique values, this is probably necessary information.

If you think you might have forgotten some details, you can always ask your interviewer to repeat the problem.

### Step 2: Draw an example

For some reason, most candidates have this instinct to stay in their chairs to solve the problem. Don’t. Get up and go to the whiteboard! It’s very difficult to solve a problem without an example.

Make sure your example is sufficiently interesting. This means that it should be not too small, but not overwhelmingly difficult, and also not a special case.

It’s surprisingly common for candidates to use a special case example. The problem with special cases is that they can make you see patterns that don’t exist, or make you fail to see patterns that do. It’s hard to distinguish between “works for this problem” and “works in general.”

### Step 3: Design a Brute Force Algorithm

As soon as you hear an interview question, try to get a solution out there, even if it’s imperfect. You can work with a brute force algorithm to optimize it.

When designing your algorithm, don’t forget to think about:

- What are the time and space complexities?
- What happen if there is a lot of data?
- Does your design cause other issues for other operations like insert/search/delete?
- If there are other issues, did you make the right trade-offs?
- If the interviewer gave you specific data, have you leveraged that information? There’s probably a reason that you’re given it.
- Even a bad solution is better than no solution. State your bad solution and then state the issues with it.

### Step 4: Optimize the Brute Force

Once you have a solution out there, focus on making it better.

If you have a brute force algorithm, it often works well to run through the algorithm—by hand with your example, not by writing code—and look for areas to optimize. Specifically, look for the bottlenecks, unnecessary work, and duplicated work areas:

- If there one part of the code that’s taking a long time? For example, if your algorithm has first step that’s O(N log N) and a second step that’s O(N), there’s little sense in optimizing the second step. The first step will always be a bottleneck. A bottleneck could also be a particularly slow part of the code that is called repeatedly. That might be a good place on which to focus your optimizations.
- Unnecessary work. Is there anything you’re doing that isn’t really necessary? For example, perhaps you’re searching for an element on both sides of the tree, when you should really have some clue why it would be on one side.
- Duplicated work. Is there anything you’re doing over and over again? For example, if you’re continuously searching for the same elements, this could constitute duplicated work and you could optimize it with a hash table.

Of course, if a really novel and unrelated approach comes to you, don’t be afraid to start from scratch.

### Step 5: Understand the Code

Interviewees spend too little time on this step and, unfortunately, it typically results in their writing sloppy and incorrect code.

It’s a bad habit instilled in coders from using a computer. If the code is short enough, you’re used to just typing it out and running it, then fixing up what doesn’t work. This is okay on a computer: typing a short problem is pretty fast.

On a whiteboard, though, it’s very slow to write code and even slower to test it. This is why it’s important to make sure you really, really know what you’re doing.

Run through your algorithm meticulously before coding. For example, imagine you’re trying to merge two sorted arrays into a new sorted array. Many candidates start coding when they understand the basic gist: two pointers move them through the array, copy the elements in order.

This probably isn’t sufficient. You should instead understand it deeply. You need to understand what the variables are, when they update, and why. You should have logic like this formulated before you start coding:

Initialize two pointers, p and q, which point to the beginning of A and B, respectively.

1. Initialize k to an index at the start of the result array, R.
2. Compare the values at p and q.
3. If A[p] is smaller, insert A[p] into R[k]. Increment p and k.
4. If B[q] is smaller, insert B[q] into R[k]. Increment q and k.
5. Go to step 3.

You don’t have to write this out by hand, but you do need to understand it at this level. Trying to skip a step and code before you’re totally comfortable will only slow you down.

### Step 6: Implement the Code

You don’t need to rush through your code; in fact, this will most likely hurt you. Just go at a nice, slow, methodical pace, and remember this advice.

- Use data structures generously. Where relevant, use a good data structure or define your own. For example, if you’re asked a problem involving finding the minimum age for group of people, consider defining a data structure to represent a person. This shows your interviewer that you care about good object-oriented design.
- Modularize you code first. If there are discrete steps in your algorithm, move these into separate functions. In fact, this can actually help you get out of doing tedious work. Imagine, as part of a broader algorithm, you need to convert a letter from A to Z to a number from 0 to 26. This is a tedious thing to write. Just modularize it off to another function and you probably won’t need to worry about ever writing it.
- Don’t crowd your code. Many candidates will start writing their code in the middle of the whiteboard. This is fine for the first few lines, but whiteboards aren’t that big.

If you feel yourself getting confused while coding, stop and go back to your example. You don’t need to code straight through. It’s far better that you take a break than write nonsensical code.

### Step 7: Test

It is rare for a candidate to write flawless code. Not testing therefore suggests two problems. First, it leaves bugs in your code. Second, it suggests that you’re the type of person who doesn’t test their code well.

Therefore, it’s very important to test your code.

To discover bugs the faster, do the following five steps:

1. Review your code conceptually. What is the meaning of each line? Does it do what you think it should?
2. Review error hot spots. Is there anything in your code that looks funny? Do your boundary conditions look right? What about your base cases?
3. Test against a small example. You want your example to create an algorithm to be big, but now you want a small one. An example that’s too big will take a long time to run through. This is time-consuming, but it might also cause you to rush the testing and miss serious bugs.
4. Pinpoint potential issues. What sorts of test cases would test against specific potential issues? For example, you might sense that there could be a bug with one array that’s much shorter than the other; test for this situation specifically.
5. Test error cases. Finally, test the true error conditions. What happens on a null string or negative values?


When you find a mistake, relax. Almost no one writes bug-free code; what’s important is how you react to it. Point out the mistake, and carefully analyze why the bug is occurring. Is it really just when you pass in 0, or does it happen in other cases, too?

Bugs are not a big deal. The important thing is that you think through how to fix issues you see rather than making a quick and dirty fix. A fix that works for that test case might not work for all test cases, so make sure it’s the right one.

