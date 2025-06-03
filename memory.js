import * as dotenv from 'dotenv';
dotenv.config();

import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import { ChatPromptTemplate, MessagesPlaceholder } from '@langchain/core/prompts';

import { ConversationChain } from 'langchain/chains'

//Memory Imports
import { BufferMemory } from 'langchain/memory';
import { UpstashRedisChatMessageHistory } from '@langchain/community/stores/message/upstash_redis';
import { RunnableSequence } from '@langchain/core/runnables';

const model = new ChatGoogleGenerativeAI({
    model: 'gemini-1.5-flash',
    temperature: 0.7,
    apiKey: process.env.GOOGLE_GENAI_API_KEY,
});

const prompt = ChatPromptTemplate.fromTemplate(`
    You are an AI assistant.
    History: {history}
    {input}
    `);


const upstashChatHistory = new UpstashRedisChatMessageHistory({
    sessionId: 'chat1',// unique session id used to restart the conversation(can generate using any unique id generator)
    config: {
        url: process.env.UPSTASH_REDIS_REST_URL,
        token: process.env.UPSTASH_REDIS_REST_TOKEN,
    }
});

const memory = new BufferMemory({
    memoryKey: 'history',
    chatHistory: upstashChatHistory,
});

// Using Chain classes
// const chain = new ConversationChain({
//     llm: model,
//     prompt,
//     memory,
// });

//Using LCEL - Langchain Expression Language
// const chain = prompt.pipe(model);
const chain = RunnableSequence.from([
    {
        input: (initialInput) => initialInput.input,
        memory: () => memory.loadMemoryVariables(),
    },
    {
        input: (previousOutput) => previousOutput.input,
        history: (previousOutput) => previousOutput.memory.history,
    },
    prompt,
    model
]);




//get responses
// console.log(await memory.loadMemoryVariables());

// const input1 = {
//     input: 'The passphrase is KESHINCOMMANDER'
// }
// const resp1 = await chain.invoke(input1);
// console.log(resp1);

// // manually updating buffer memory for RunnableSequence
// await memory.saveContext(input1, {
//     output: resp1.content
// });

// console.log("Updated History:", await memory.loadMemoryVariables());

const input2 = {
    input: 'What is the passphrase?'
}
const resp2 = await chain.invoke(input2);
console.log(resp2);

// manually updating buffer memory for RunnableSequence
await memory.saveContext(input2, {
    output: resp2.content
});
