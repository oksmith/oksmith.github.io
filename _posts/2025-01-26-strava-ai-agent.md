---
layout: post
title: "Building an AI Agent to manage my Strava activities"
date: 2025-01-26
categories: 
  - "machine-learning"
tags: 
  - "activity"
  - "agent"
  - "ai"
  - "bot"
  - "langchain"
  - "langgraph"
  - "llm"
  - "running"
  - "strava"
coverImage: "/images/run-scaled.jpg"
published: true
---

![](/images/run-1024x599.jpg)
My phone took this photo without me realising whilst on a run
{:style="color:gray;font-style:italic;font-size:90%;text-align:center;"}

In this post I wanted to write down some of my learnings when building an application to update my Strava activities with fun poems.

### What is an agent?

Different people seem to define an "agent" in different ways. But I like this one by Anthropic \[1\]:

- **Workflows** are systems where LLMs and tools are orchestrated through predefined code paths.

- **Agents**, on the other hand, are systems where LLMs _dynamically direct their own processes and tool usage_, maintaining control over how they accomplish tasks.

The key here being the _agency_ with which the agents can select which function to use and how to use it.

### The Strava agent

I wanted to get to know what current frameworks to build agentic applications are like. I also really enjoy running, but never really add a description to any of my runs on Strava. I also noticed that Strava allows developers [API access](https://developers.strava.com/docs/reference/). These were the three ingredients to the Strava bot idea!

![](/images/graph-1024x568.png)
The structure of my Strava bot. There are a set of tools, one of which requires human intervention before calling.
{:style="color:gray;font-style:italic;font-size:90%;text-align:center;"}


#### LangGraph

I decided to use LangGraph for my agent. This is because LangChain is one of the most popular frameworks for building generative AI applications out there, and as of [LangChain v0.3](https://python.langchain.com/docs/how_to/migrate_agent/), the recommended way of building agents is through LangGraph rather than the legacy AgentExecutor in LangChain.

In every LangGraph agents there are nodes and edges - much like a normal graph. There is always at least one chatbot note, which is an LLM that reasons with the information it is given and orchestrates the rest of the application.

There are also "tool" nodes. These are essentially Python functions, with docstrings and clearly defined input and output schemas, which the LLM is given context about. Through LangGraph, the LLM is able to execute such functions to obtain additional data that it needs in order to resolve the user query. It can execute them in any order it thinks is the most appropriate, and it does all this using its natural language understanding and the context you provide about the functions themselves.

In the diagram above you'll see solid lines and dotted lines. The dotted lines represent _conditional edges_, where the LLM may decide to move to that node, and solid lines are normal edges, meaning that once the process on that node has completed, it _must_ now move down that edge to the next task.

In LangGraph you can also specify _memory_ via checkpointers. This allows the chatbot to keep track of the current conversation, so that it can refer back to calculations it has already done. In my case, if the bot has already fetched Google maps data, then if the user asks another question related to location, it should know that it doesn't need to fetch Google maps data second time (as it's already in the context / memory saver).

And finally, there is a human interruption component. This is where, before any particularly consequential actions are executed by the LLM, I make sure a human gives permission. LangGraph has a nice interrupt / Command functionality to handle these situations.

#### How my Strava bot works

The tools I added were:

- **Fetch activities**: GET request to Strava's activities endpoint

- **Select activity**: An LLM identifies a _specific_ activity based on context. For example, if the user is referring to "my run from this morning", the select activity tool will look for activities that are (a) runs, (b) match today's date.

- **Fetch map data**: This decodes the polyline from Strava into a sequence of coordinates, which are then passed into Google Maps API to fetch nearby street names and tourist attractions on my route.

- **Generate poem**: Using the map data from the selected activity, ask the LLM to generate a poem! I've prompted all of the poems to rhyme, to be 8 lines by default, and to use street names / landmarks if available. This way, if I ran around Hyde Park, then Hyde Park should feature at some point in the poem.

- **Update activity**: Once the user is happy with the poem, they might want to update their Strava activity. This is a more "dangerous" activity than normal, so I've set up an interruption to double check the user is happy to proceed.

You can check out the code for all of this [on GitHub here](https://github.com/oksmith/running-buddy). I originally called the project "stravabot", but I changed it because Strava don't allow 3rd party applications with the word "Strava" in it.

### Usage

Here's a gif demonstrating the intended usage of "Running Buddy":

![](/images/agent.gif)

Here's the corresponding update to my run on Strava:

![](/images/strava.png)

Overall, it seems to work surprisingly well! I can have conversations with the chatbot, and iteratively refine the poem if it wasn't quite right. I could cancel the activity update if I had a change of heart, and continue with the update if I'm happy to proceed. It remembers the earlier parts of the conversation and adapts its tool usage accordingly.

Some prompt engineering went a long way; to begin with the bot didn't know it was supposed to use the external APIs, and just started making random generic poems about morning runs.

I later added a new tool which searches the current weather using DuckDuckGo, and tells me whether it thinks now's a good time to go on a run or not. There are several ways you could extend this project further to be a proper "running buddy"!

### Key learnings

This was just a fun personal project meant for understanding how these systems work. I've learned a few things which weren't immediately obvious to me before starting out:

1. **Having a human in the loop** is a seriously powerful addition to AI applications. For many of the apps you might want to build, there are probably actions you wouldn't want the AI agent to make without some kind of human supervision. The use of a human to oversee important components is a huge step in the right direction for safety and reliability. I also found it quite cool how LangGraph handles human-in-the-loop, with an `interrupt()` invocation causing the state of the graph to pause and stop streaming updates, only continuing once you pass in a `Command()` back into it, at which point it continues with where it started.  
    

3. **Structured outputs** are also a pretty awesome way to force the LLM to return data in a specific schema. In LangChain this works with a Pydantic data class that you specify. This means you can be confident that the data has been validated and is fit for use for downstream tasks. For my application, selecting a Strava activity seems to be the slowest part, so this area might be something I pay more attention to if I develop the app further.  
    

5. **There are use cases where agentic applications are more effective**. Anthropic listed a few in their article -- for example, customer support agents and coding assistants. They found that the scenarios where agents were most effective were tasks that require both conversation and action, have clear success criteria, enable feedback loops, and integrate meaningful human oversight. They also outlined some very powerful **workflows** (which are predefined systems without agency), which combine outputs of many LLMs together to solve difficult tasks effectively.  
    

7. **LangChain documentation** is hard to deal with. Lots of examples and code snippets you'll find online are for legacy versions of LangChain which don't work anymore. Because of this, I made the switch from LangChain -> LangGraph pretty early on as I wanted to stick with LangChain v0.3. Documentation seems to overall be lagging behind in this space, in large part due to the speed of development.

Also, in general, this mini project really made me realise the power of the tools you can enhance an LLM with. An LLM on its own is a powerful language reasoning engine, but can't do much beyond question-answering and conversation. An LLM with access to the internet, to external APIs, to internal data stores, to BigQuery data processing, executing some code, sending an email (and the list goes on) -- really is more like an autonomous entity that can enhance the quality of its own responses and take real actions.

### Useful resources

\[1\] Building Effective Agents (Anthropic): [https://www.anthropic.com/research/building-effective-agents](https://www.anthropic.com/research/building-effective-agents)

\[2\] LangGraph docs: [https://langchain-ai.github.io/langgraph/](https://langchain-ai.github.io/langgraph/)
