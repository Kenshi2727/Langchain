import * as dotenv from 'dotenv';
dotenv.config();
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import { ChatPromptTemplate, MessagesPlaceholder } from '@langchain/core/prompts';
import { createToolCallingAgent, AgentExecutor } from 'langchain/agents';
import { TavilySearch } from '@langchain/tavily';
import { HumanMessage, AIMessage } from '@langchain/core/messages';
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter"

import { createRetrieverTool } from 'langchain/tools/retriever'

import readline, { createInterface } from 'readline';


/*LOAD DATA FROM WEBPAGE */
// creating loader
const loader = new CheerioWebBaseLoader("https://en.wikipedia.org/wiki/Anime");
const docs = await loader.load();
// console.log(docs);

// splitting the documents into smaller chunks
const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 200,
    chunkOverlap: 20,
});

// splitting the documents
const splitDocs = await splitter.splitDocuments(docs);
// console.log(splitDocs);

//instantiating the embeddings
const embeddings = new GoogleGenerativeAIEmbeddings({
    apiKey: process.env.GOOGLE_GENAI_API_KEY,
});

//creating in-memory vector store
const vectorStore = await MemoryVectorStore.fromDocuments(splitDocs, embeddings);


//retrieving the documents from the vector store

//setting up a retriever
const retriever = vectorStore.asRetriever({
    k: 2, // number of documents to retrieve
});



const model = new ChatGoogleGenerativeAI({
    model: 'gemini-1.5-flash',
    temperature: 0.7,
    apiKey: process.env.GOOGLE_GENAI_API_KEY,
});

const prompt = ChatPromptTemplate.fromMessages([
    ['system',

        `You are a helpful assistant called Ramu.You have access to a 
    tool named "TavilySearch" which you should use to answer questions about 
    real-time information such as current weather, news, or facts.When you 
    need information beyond your knowledge, call the TavilySearch tool.
    For anime you can use the "Anime_Search" tool to search for information about animes.`
    ],

    new MessagesPlaceholder('chat_history'),

    ['human', '{input}'],
    new MessagesPlaceholder('agent_scratchpad'),
]);

//create and assign tools
const searchTool = new TavilySearch();
// create retriever tool
const retrieverTool = createRetrieverTool(retriever, {
    name: "Anime_Search",// name of the tool
    description: "Use this tool when searching for information about animes",//tells the agent when to use this tool
});

const tools = [searchTool, retrieverTool];

// creating agent
const agent = await createToolCallingAgent({
    llm: model,
    prompt,
    tools,
});

//creating agent executor for invoking the agent
const agentExecutor = new AgentExecutor({
    agent,
    tools,
});

// // calling agent
// const response = await agentExecutor.invoke({
//     input: "what is the current weather in New York?",
// });

// console.log(response);

//get user input
const rl = createInterface({
    input: process.stdin,
    output: process.stdout,
});

const chatHistory = [];

const askQuestion = () => {
    rl.question("User: ", async (input) => {

        // Exit if the user types 'exit'
        if (input.toLowerCase() === 'exit') {
            console.log("Exiting...");
            rl.close();
            return;
        }

        // calling agent
        const response = await agentExecutor.invoke({
            input,
            chat_history: chatHistory,
        });

        console.log("Agent:", response.output);

        // Add the question and answer to chat history
        chatHistory.push(new HumanMessage(input));
        chatHistory.push(new AIMessage(response.output));

        // Ask the next question
        askQuestion();
    });
}

// Start the question loop
askQuestion();

