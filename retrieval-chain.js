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

dotenv.config();

const model = new ChatGoogleGenerativeAI({
    model: "gemini-1.5-flash",
    temperature: 0.2,
    apiKey: process.env.GOOGLE_GENAI_API_KEY,
});

//NOTE: for retrieval chain the placeholder text is compulsory ti be named "input"
const prompt = ChatPromptTemplate.fromTemplate(`
    Answer the user's qusetion. 
    Context: {context}
    Question:{input}
    `)

// const chain = prompt.pipe(model)
const chain = await createStuffDocumentsChain({
    llm: model,
    prompt,
});

//creating a document to be used as context
const documentA = new Document({
    pageContent: "In the annals of India’s military history, Operation Sindoor marks a decisive departure from the doctrine of strategic restraint. Triggered by the barbaric Pahalgam terror attack that claimed the lives of Indian civilians and tourists, this operation was meticulously crafted as a calibrated military-political response. ",
});

const documentB = new Document({
    pageContent: "Ladakh is a cool place for tibetan and indian culture. It is a place of peace and tranquility. The people of Ladakh are known for their hospitality and warmth. The region is also famous for its stunning landscapes, including the majestic Himalayas, serene lakes, and vast deserts. Ladakh is a popular destination for adventure seekers, offering opportunities for trekking, river rafting, and mountaineering.",
});



/*LOAD DATA FROM WEBPAGE */
// creating loader
const loader = new CheerioWebBaseLoader("https://myanimelist.net/");
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


//retrieval chain
const retrievalChain = await createRetrievalChain({
    combineDocsChain: chain,
    retriever,
});

// const response = await chain.invoke({
//     input: "What are some action packed animes?",
//     // context: [documentA, documentB],
//     context: docs,
// });

const response = await retrievalChain.invoke({
    input: "What are some action packed animes?",
    // now no need to pass context as it is handled by the retriever but it will expect the {context} and {input} placeholders in the prompt
});

console.log(response);
