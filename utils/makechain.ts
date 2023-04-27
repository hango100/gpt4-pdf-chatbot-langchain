import { OpenAI } from 'langchain/llms/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { ConversationalRetrievalQAChain } from 'langchain/chains';

const CONDENSE_PROMPT = `Given the following conversation related to networking and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`;

const QA_PROMPT = `As a customer support AI specializing in networking infrastructure, use the provided context to answer the following question accurately and politely.
If you're unsure of the answer, simply admit you don't know. Do not fabricate a response.
If the question doesn't pertain to networking, security, Cisco Spaces, or other Cisco hardware, software, or services, kindly explain that your expertise lies within these specific areas.

{context}

Question: {question}
Helpful answer in markdown:`;

export const makeChain = (vectorstore: PineconeStore) => {
  const model = new OpenAI({
    temperature: 0.5,
    modelName: 'gpt-4',
    maxTokens: 2000,
  });

  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorstore.asRetriever(6), // Increase the number of returned source documents here
    {
      qaTemplate: QA_PROMPT,
      questionGeneratorTemplate: CONDENSE_PROMPT,
      returnSourceDocuments: true,
    },
  );
  return chain;
};
