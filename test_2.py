import streamlit as st
import os
from dotenv import load_dotenv
from py2neo import Graph
from langchain.chains import LLMChain
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

# Access the API key
groq_api_key = os.getenv("GROQ_API_KEY")

# Neo4j Graph Database Connection
graph = Graph("bolt://localhost:7687", auth=("neo4j", "Password"))

# Function to retrieve course recommendations based on user interests
def get_course_recommendation(interests):
    query = """
    MATCH (c:Course)
    WHERE c.name IN $interests
    RETURN c.name AS course_name, c.duration AS duration, c.time AS time, c.fees AS fees
    LIMIT 1
    """
    result = graph.run(query, interests=interests).data()
    return result[0] if result else None

def main():
    # Title and greeting message
    st.title("Chat with ZoroBot!")
    st.write("Hello! I'm Zoro, your friendly career course advisor. Let's chat about your interests like 'Data Science,' 'Python,' or 'Flutter,' and I'll recommend the best courses for you!")

    # Initialize session state for chat history if it doesn't exist
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Initialize memory for the conversation
    memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)

    # System prompt to guide the chatbot's responses
    system_prompt = """
    You are an educational chatbot guiding students toward suitable career-oriented courses. Your responses should:
    1. Refer to previous conversation topics to ensure relevance.
    2. Incorporate details from the conversation history to maintain coherence.
    3. Avoid redundant responses, and address new queries directly based on existing context.
    4. Provide information concisely, using clear language unless technical details are necessary.
    Respond only with verified information available to you. If the answer isnot known, reply with 'I am unable to provide information on that.' Do not fabricate details, guess, or assume beyond provided data. 
    """

    user_question = st.text_input("Ask a question or share your interest:")

    # Add previous chat history to memory
    for message in st.session_state.chat_history:
        memory.save_context({'input': message['human']}, {'output': message['AI']})

    # Initialize the Groq chat object
    groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name='llama3-8b-8192')

    # If the user has asked a question,
    if user_question:
        # Construct a chat prompt template using various components
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{human_input}"),
            ]
        )

        # Create a conversation chain using the LangChain LLM (Language Learning Model)
        conversation = LLMChain(
            llm=groq_chat,
            prompt=prompt,
            verbose=True,
            memory=memory,
        )

        # Generate a response based on user input
        response = conversation.predict(human_input=user_question)

        # Save the current question and response to session history
        message = {'human': user_question, 'AI': response}
        st.session_state.chat_history.append(message)

        # Capture user interests for course recommendations
        user_interests = set()
        for message in st.session_state.chat_history:
            if "human" in message:
                content = message["human"].lower()
                if "python" in content:
                    user_interests.add("Python")
                elif "data science" in content:
                    user_interests.add("Data Science")
                elif "flutter" in content:
                    user_interests.add("Flutter")
                elif "mern stack" in content:
                    user_interests.add("MERN Stack")

        # Generate a relevant course recommendation if interests are identified
        if user_interests:
            course_info = get_course_recommendation(list(user_interests))
            if course_info:
                response += f"\n\nBased on your interests, I recommend the **{course_info['course_name']}** course:\n" \
                            f"- **Duration**: {course_info['duration']}\n" \
                            f"- **Time Commitment**: {course_info['time']} daily\n" \
                            f"- **Fees**: {course_info['fees']}\n" \
                            "If this aligns with your goals, I can share further details or suggest alternatives."
            else:
                response += "\n\nSorry, I couldn't find an exact match. Could you share more details or specify other interests?"

        # Display the response to the user
        st.write("Zoro:", response)

if __name__ == "__main__":
    main()
