import os
import json
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

class SarkariSevaAssistant:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile")
        self.document_storage = "user_documents.json"
        if not os.path.exists(self.document_storage):
            with open(self.document_storage, "w") as f:
                json.dump({}, f)

    def store_document(self, user_id, document_name, document_data):
        with open(self.document_storage, "r") as f:
            documents = json.load(f)
        if user_id not in documents:
            documents[user_id] = {}
        documents[user_id][document_name] = document_data
        with open(self.document_storage, "w") as f:
            json.dump(documents, f)
        return f"Document '{document_name}' stored successfully for user {user_id}."

    def get_document(self, user_id, document_name):
        with open(self.document_storage, "r") as f:
            documents = json.load(f)
        if user_id in documents and document_name in documents[user_id]:
            return documents[user_id][document_name]
        return f"Document '{document_name}' not found for user {user_id}."

    def clarify_service_name(self, service_name):
        prompt_clarify = PromptTemplate.from_template(
            """
            ### USER INPUT:
            {service_name}

            ### INSTRUCTION:
            You are a helpful assistant for SarkariSeva, a government service portal.
            The user has provided a service name, but it might be unclear or incomplete.
            Your task is to rephrase it in simpler terms and confirm with the user if this is what they meant.
            If the input is already clear, just repeat it as is.
            ### CLARIFIED SERVICE NAME:
            """
        )
        chain_clarify = prompt_clarify | self.llm | StrOutputParser()
        clarified_name = chain_clarify.invoke({"service_name": service_name})
        return clarified_name

    def generate_steps(self, service_name):
        prompt_steps = PromptTemplate.from_template(
            """
            ### SERVICE NAME:
            {service_name}

            ### INSTRUCTION:
            You are a helpful assistant for SarkariSeva, a government service portal.
            Provide a simple, step-by-step guide in Hinglish for applying to the service mentioned above.
            Include all the necessary documents and information required.
            Keep the instructions clear and easy to follow.
            ### STEPS (IN HINGLISH):
            """
        )
        chain_steps = prompt_steps | self.llm | StrOutputParser()
        steps = chain_steps.invoke({"service_name": service_name})
        return steps

    def answer_question(self, question):
        prompt_question = PromptTemplate.from_template(
            """
            ### USER QUESTION:
            {question}

            ### INSTRUCTION:
            You are a helpful assistant for SarkariSeva, a government service portal.
            Answer the user's question in clear and simple Hinglish.
            If the question is about a specific service, provide relevant information or steps.
            ### ANSWER (IN HINGLISH):
            """
        )
        chain_question = prompt_question | self.llm | StrOutputParser()
        answer = chain_question.invoke({"question": question})
        return answer

    def apply_for_service(self, user_id, service_name):
        clarified_name = self.clarify_service_name(service_name)
        print(f"Did you mean: {clarified_name}? (Yes/No)")
        user_confirmation = input().strip().lower()
        if user_confirmation != "yes":
            return "Please provide the correct service name."
        steps = self.generate_steps(clarified_name)
        required_docs = self.answer_question(f"{clarified_name} ke liye kaunse documents chahiye?")
        return {
            "steps": steps,
            "required_documents": required_docs,
            "message": f"Follow these steps to apply for {clarified_name}. Required documents: {required_docs}"
        }

if __name__ == "__main__":
    assistant = SarkariSevaAssistant()

    user_id = "user123"
    service_name = input("Enter the service you need help with: ").strip()
    print(assistant.apply_for_service(user_id, service_name))
