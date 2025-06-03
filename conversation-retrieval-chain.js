import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { Document } from "langchain/document";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter"
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import * as dotenv from "dotenv";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import { MessagesPlaceholder } from "@langchain/core/prompts";
import { createHistoryAwareRetriever } from 'langchain/chains/history_aware_retriever';

dotenv.config();

// load data and crate vector store
const createVectorStore = async () => {
    const loader = new CheerioWebBaseLoader("https://myanimelist.net/");
    const docs = await loader.load();

    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 200,
        chunkOverlap: 20,
    });


    const splitDocs = await splitter.splitDocuments(docs);


    const embeddings = new GoogleGenerativeAIEmbeddings({
        apiKey: process.env.GOOGLE_GENAI_API_KEY,
    });


    const vectorStore = await MemoryVectorStore.fromDocuments(splitDocs, embeddings);

    return vectorStore;
}


// create retrieval chain
const createChain = async () => {

    const model = new ChatGoogleGenerativeAI({
        model: "gemini-1.5-flash",
        temperature: 0.2,
        apiKey: process.env.GOOGLE_GENAI_API_KEY,
    });

    // const prompt = ChatPromptTemplate.fromTemplate(`
    // Answer the user's qusetion. 
    // Context: {context}
    // Chat History: {chat_history}
    // Question:{input}
    // `)


    /////////////////////////////////////////////////////////////////////////////////////////
    /*since chat_history placheolder expects a text not an array of messages, we need to use 
    fromMessages to convert the chat history to a text format*/
    ////////////////////////////////////////////////////////////////////////////////////////

    const prompt = ChatPromptTemplate.fromMessages([
        ["system", "Anser the question based on the context provided: {context}"],
        // accepts an array of messages and converts it into string
        new MessagesPlaceholder("chat_history"),
        ["user", "{input}"],
    ]);


    const chain = await createStuffDocumentsChain({
        llm: model,
        prompt,
    });

    const retriever = vectorStore.asRetriever({
        k: 2, // number of documents to retrieve
    });


    const retrieverPrompt = ChatPromptTemplate.fromMessages([
        new MessagesPlaceholder("chat_history"),
        ["user", "{input}"],
        [
            "user",
            "Given the above conerstaion, generate a serch query to look up in order get information relevant to the conversation."
        ],
    ]);

    // create a history aware retriever
    const historyAwareRetriever = await createHistoryAwareRetriever({
        llm: model,
        retriever,
        rephrasePrompt: retrieverPrompt,
    });


    // const retrievalChain = await createRetrievalChain({
    const conversationChain = await createRetrievalChain({
        combineDocsChain: chain,
        retriever: historyAwareRetriever,
    });

    return conversationChain;

}







const vectorStore = await createVectorStore();
const chain = await createChain(vectorStore);



//chat  History - // This is an example of how you can use chat history to provide context to the model
const chatHistory = [
    new HumanMessage("What are some action packed animes?"),
    new AIMessage("Some action-packed animes include Attack on Titan, Demon Slayer, and My Hero Academia."),
    new HumanMessage("What about some romantic animes?"),
    new AIMessage("Some romantic animes include Your Lie in April, Toradora!, and Clannad."),
    new HumanMessage("Should ecchis be banned?"),
    new AIMessage("Ecchi is a genre that some people enjoy, while others may find it inappropriate. It's a matter of personal preference and cultural context."),
];



const response = await chain.invoke({
    input: "What is that inappropriate??",
    chat_history: chatHistory,
});

console.log(response);
