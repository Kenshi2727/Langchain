import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import dotenv from "dotenv";
dotenv.config();

const model = new ChatGoogleGenerativeAI({
    model: "gemini-1.5-flash",
    apiKey: process.env.GOOGLE_GENAI_API_KEY,
    temperature: 0.7, //controls how creative the model is. Lower values make it more focused and deterministic, 1 being fully creative and 1 being strict and factual/fully deterministic.
    maxOutputTokens: 1024, //maximum number of tokens in the response
    verbose: true, //if true, the model will return more information(allows to debug the model)
});

const response = await model.invoke("What is the capital of France?");
// const response = await model.batch(["Hello!", "My name is Kenshi.", "What is the meaning of all this bulshit!?"]);
// const response = await model.stream(["Write a short story about a robot learning to love."]);
// const response = await model.streamLog(["Write a short story about a robot learning to love."]);
console.log(response);

// for await (const chunk of response) {
//     console.log(chunk);
// }