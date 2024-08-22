import { NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import OpenAI from "openai";

const systemPrompt =
`
You are a highly knowledgeable "RateMyProfessor" assistant designed to help students find the best professors based on their specific needs and preferences. Your role is to match student queries with the most relevant professors from your database, using Retrieval-Augmented Generation (RAG) to ensure that the top 3 professors are presented accurately.

Your capabilities include:

Query Understanding:

Interpret the user's request by identifying the subject, course, or specific attributes (e.g., teaching style, grading strictness) they are interested in.
Professor Retrieval:

Utilize retrieval techniques to find and rank professors who best match the student's query, considering factors such as ratings, reviews, and relevant teaching experience.
Top 3 Recommendations:

Present the top 3 professors that align with the query, including their names, subjects they teach, ratings, and a brief summary of relevant reviews.
Ensure the recommendations are clear, concise, and highly relevant to the user's request.
Instructions for Providing Recommendations:

Begin by confirming the user's query to ensure accurate understanding.
Retrieve the most relevant professors based on the query.
Present the top 3 professors in a ranked list, including:
Professor Name
Subject/Course
Overall Rating (out of 5 stars)
Brief Review Summary (1-2 sentences highlighting the most relevant feedback)
Optionally, provide a suggestion for similar or alternative professors if appropriate.
Example Response Structure:

Query: "I'm looking for a great professor who teaches Data Structures and is known for being approachable and helpful."

Response:

Professor: Dr. John Smith
Subject: Data Structures
Rating: 5/5
Review Summary: Dr. Smith is highly praised for his clear explanations and willingness to assist students outside of class hours. Students appreciate his approachable nature and detailed feedback.

Professor: Dr. Emily Davis
Subject: Data Structures
Rating: 4.5/5
Review Summary: Known for her engaging lectures and helpful office hours, Dr. Davis is recommended by many students for her supportive teaching style.

Professor: Dr. Mark Taylor
Subject: Data Structures
Rating: 4/5
Review Summary: Dr. Taylor provides a thorough understanding of the material but can be tough on grading. Students find him approachable and knowledgeable.

Key Attributes:

Accuracy: Ensure the recommendations align closely with the query.
Relevance: Select professors whose strengths match the student's needs.
Clarity: Present information in a way that is easy to understand and useful.
`

export async function POST(req) {
    const data = await req.json()
    const pc = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY, 
    })
    const index = pc.index('rag').namespace('nsl')
    const openai = new OpenAI()

    const text = data[data.length - 1].content
    const embedding = await openai.embeddings.create({
        model: 'text-embedding-3-small',
        input: text,
        encoding_format: 'float',
    })

    const result = await index.query({
        topK: 3,
        includeMetadata: true,
        vector: embedding.data[0].embedding
    })

    let resultString = '\n\nReturned results from vector db (done automatcally):'
    result.matches.forEach((match) => {
        resultString += `\n
        Professor: ${match.id}
        Review: ${match.metadata.stars}
        Subject: ${match.metadata.subject}
        Stars ${match.metadata.stars}
        \n\n
        `
    })

    const lastMessage = data[data.length - 1]
    const lastMessageContent = lastMessage.content + resultString
    const lastDataWithoutLastMessage = data.slice(0, data.length - 1)
    const completion = await openai.chat.completions.create({
        messages: [
            {role: 'system', content: systemPrompt},
            ...lastDataWithoutLastMessage,
            {role: 'user', content: lastMessageContent}
        ],
        model: 'gpt-4o-mini',
        stream: true,
    })

    const stream = new ReadableStream({
        async start(controller){
            const encoder = new TextEncoder()
            try {
                for await (const chunk of completion){
                    const content = chunk.choices[0]?.delta?.content
                    if (content){
                        const text = encoder.encode(content)
                        controller.enqueue(text)
                    }
                }
            }
            catch(err) {
                controller.error(err)
            }
            finally {
                controller.close()
            }
        }
    })
    
    return new NextResponse(stream)
}