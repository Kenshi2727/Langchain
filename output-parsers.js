import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import * as dotenv from "dotenv";
import { StringOutputParser, CommaSeparatedListOutputParser, StructuredOutputParser } from "@langchain/core/output_parsers";
import { z } from "zod";

dotenv.config();


const model = new ChatGoogleGenerativeAI({
    apiKey: process.env.GOOGLE_GENAI_API_KEY,
    model: "gemini-1.5-flash",
    temperature: 0.7,
});

// convert the output of the model to a string 
async function callStringOutputParser() {

    // Create a string output parser
    const parser = new StringOutputParser();

    const prompt = ChatPromptTemplate.fromMessages([
        ["system", "Generate a joke based on a word provided by the user."],
        ["user", "{input}"]
    ]);

    // attahing the output parser to the prompt
    const chain = prompt.pipe(model).pipe(parser);
    return await chain.invoke(
        { input: "dog" }
    );

}

// convert string to a javascrpit array of strings
async function callListOutputParser() {
    const prompt = ChatPromptTemplate.fromTemplate(`
        Provide five synonyms, separated by commas, for the word {word}.
    `);

    const parser = new CommaSeparatedListOutputParser();

    const chain = prompt.pipe(model).pipe(parser);

    return await chain.invoke({ word: "happy" });
}

// convert the output of the model to a JSON object
// response type- js object
async function callStructuredOutputParser() {
    const prompt = ChatPromptTemplate.fromTemplate(`
    Extract information from the following pharse.
    Formatting Instructions: {format_instructions}
    Phrase: {phrase}
    `);

    // defining formatting instructions for the output parser
    const parser = StructuredOutputParser.fromNamesAndDescriptions({
        name: "The name of the person",
        age: "The age of the person",
    });

    const chain = prompt.pipe(model).pipe(parser);
    return await chain.invoke({
        phrase: "Kenshi is 22 years old.",
        format_instructions: parser.getFormatInstructions(),
    });
}

// using zod with structured output parser
// This allows you to define a schema for the output and validate it
async function callZodOutputParser() {
    const prompt = ChatPromptTemplate.fromTemplate(`
    Extract information from the following pharse.
    Formatting Instructions: {format_instructions}
    Phrase: {phrase}
    `);

    // object schema for the output parser
    const parser = StructuredOutputParser.fromZodSchema(
        z.object({
            recepie: z.string().describe("The name of the recepie"),
            ingredients: z.array(z.string()).describe("The ingredients of the recepie"),
            // array with content type string 
        })
    );

    const chain = prompt.pipe(model).pipe(parser);
    return await chain.invoke({
        phrase: "The ingredients for maggie noodles are water, noodles, tastemaker and vegetables.",
        format_instructions: parser.getFormatInstructions(),
    })
}

// const response = await callStringOutputParser();
// const response = await callListOutputParser();
// const response = await callStructuredOutputParser();
const response = await callZodOutputParser();
console.log(response);


