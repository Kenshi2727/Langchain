import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatPromptTemplate } from "@langchain/core/prompts";

import * as dotenv from "dotenv";
dotenv.config();

//create model
const model = new ChatGoogleGenerativeAI({
    apiKey: process.env.GOOGLE_GENAI_API_KEY,
    model: "gemini-1.5-flash",
    temperature: 0.7,
});

//create prompt template
// const prompt = ChatPromptTemplate.fromTemplate('You are a comedian. Tell a joke based on the following topic: {input}');
const prompt = ChatPromptTemplate.fromMessages([
    ["system", "Generate a joke based on a word provided by the user."],
    ["user", "{input}"]
]);

// console.log(await prompt.format({ input: "cats" }));

// create a chain
const chain = prompt.pipe(model);

// call chain
const response = await chain.invoke(
    { input: "cats" }
);

console.log(response);

