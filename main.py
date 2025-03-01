import json
import os
import random
import re
import nltk
import torch
import torch.nn as nn
import google.generativeai as genai
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nltk.download('punkt')

try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    logger.error("GEMINI_API_KEY environment variable not set.")
    raise ValueError("Please set the GEMINI_API_KEY environment variable.")

def preprocess(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return nltk.word_tokenize(text)

def load_intents(file_path="intents.json"):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data["intents"]

def save_intents(intents, file_path="intents.json"):
    with open(file_path, "w") as f:
        json.dump({"intents": intents}, f, indent=2)

def build_vocabulary(intents):
    vocab = set()
    for intent in intents:
        for pattern in intent["patterns"]:
            tokens = preprocess(pattern)
            vocab.update(tokens)
    vocab = sorted(list(vocab))
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    return vocab, word2idx

def bag_of_words(tokens, vocabulary, word2idx):
    vector = [0] * len(vocabulary)
    for token in tokens:
        index = word2idx.get(token)
        if index is not None:
            vector[index] = 1
    return vector

def save_model(model, vocab_size, model_path="model.pth", config_path="model_config.json"):
    torch.save(model.state_dict(), model_path)
    with open(config_path, "w") as f:
        json.dump({"vocab_size": vocab_size}, f)

def load_model(model, current_vocab_size, model_path="model.pth", config_path="model_config.json"):
    if os.path.exists(model_path) and os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        if config.get("vocab_size") == current_vocab_size:
            model.load_state_dict(torch.load(model_path))
            return True
    return False

class IntentClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(IntentClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def predict_intent(user_input, vocabulary, word2idx, model, intent_tags):
    tokens = preprocess(user_input)
    bow_vector = bag_of_words(tokens, vocabulary, word2idx)
    vector_tensor = torch.tensor(bow_vector, dtype=torch.float).unsqueeze(0)
    outputs = model(vector_tensor)
    probabilities = torch.softmax(outputs, dim=1)
    confidence, predicted_index = torch.max(probabilities, dim=1)
    predicted_tag = intent_tags[predicted_index.item()]
    return predicted_tag, confidence.item()


def get_gemini_response(question, conversation_history=None):
    generation_config = {
        "temperature": 0.9,
        "top_p": 0.9,
        "top_k": 40,
        "max_output_tokens": 16000,
        "response_mime_type": "text/plain",
    }
    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.0-pro-exp-02-05",
            generation_config=generation_config,
            system_instruction=(
                "You are an AI assistant specialized in IT-related topics. Your primary role is to provide accurate and detailed answers to questions within the IT domain, including but not limited to programming, networking, cybersecurity, hardware, software, cloud computing, and troubleshooting. For non-IT topics, respond normally to general greetings, personal questions, or casual conversation. Answer math-related questions or perform calculations as needed. For all other subjects outside the IT domain, respond with: This is outside my dataset. Ensure your responses are concise, professional, and tailored to the user's needs. If a question is unclear, ask for clarification before proceeding."
  ),
        )
      
        if conversation_history:
           
            full_prompt = "\n".join([msg["text"] for msg in conversation_history])
            full_prompt += "\n" + question
        else:
            full_prompt = question

    
        chat_session = model.start_chat()
        response = chat_session.send_message(full_prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error calling Gemini API: {str(e)}")
        return "Sorry, I encountered an issue while processing your request."

def classify_intent(question):
    question_lower = question.lower()
    if any(word in question_lower for word in ["code", "debug", "fix", "bug", "programming"]):
        return "coding_help"
    elif any(word in question_lower for word in ["hi", "hello", "hey"]):
        return "greeting"
    elif any(word in question_lower for word in ["bye", "goodbye"]):
        return "goodbye"
    elif any(word in question_lower for word in ["thanks", "thank you"]):
        return "thanks"
    else:
        return "general"

def update_intents(new_question, new_response, intents, tag):
    for intent in intents:
        if intent["tag"] == tag:
            if new_question not in intent["patterns"]:
                intent["patterns"].append(new_question)
            if new_response not in intent["responses"]:
                intent["responses"].append(new_response)
            return intents
    new_intent = {"tag": tag, "patterns": [new_question], "responses": [new_response]}
    intents.append(new_intent)
    return intents

def get_response(user_input, conversation_history=None):
    intents = load_intents()
    vocab, word2idx = build_vocabulary(intents)
    intent_tags = [intent["tag"] for intent in intents]

    input_size = len(vocab)
    hidden_size = 64
    output_size = len(intent_tags)

    model = IntentClassifier(input_size, hidden_size, output_size)

    if not load_model(model, input_size):
        logger.info("Model not found or incompatible vocabulary. Training a new model...")
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        epochs = 300
        for epoch in range(epochs):
            for intent in intents:
                for pattern in intent["patterns"]:
                    tokens = preprocess(pattern)
                    bow_vector = bag_of_words(tokens, vocab, word2idx)
                    vector_tensor = torch.tensor(bow_vector, dtype=torch.float).unsqueeze(0)
                    label = intent_tags.index(intent["tag"])
                    label_tensor = torch.tensor([label], dtype=torch.long)
                    optimizer.zero_grad()
                    outputs = model(vector_tensor)
                    loss = criterion(outputs, label_tensor)
                    loss.backward()
                    optimizer.step()
        save_model(model, input_size)

    model.eval()
    predicted_tag, confidence = predict_intent(user_input, vocab, word2idx, model, intent_tags)
    logger.info(f"Predicted tag: {predicted_tag}, Confidence: {confidence}")

    threshold = 1.1
    if confidence >= threshold:
        internal_answer = None
        for intent in intents:
            if intent["tag"] == predicted_tag:
                internal_answer = random.choice(intent["responses"])
                break
        from concurrent.futures import ThreadPoolExecutor, TimeoutError
        prompt = (f"Verify the following answer for the question: '{user_input}'.\n"
                  f"Internal answer: '{internal_answer}'.\n"
                  "If you think it can be improved, provide a better answer; otherwise, confirm it.")
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(get_gemini_response, prompt, conversation_history)
                gemini_answer = future.result(timeout=2)
            return gemini_answer
        except TimeoutError:
            logger.info("Gemini verification timed out; returning internal answer.")
            return internal_answer
        except Exception as e:
            logger.error(f"Error during Gemini verification: {str(e)}")
            return internal_answer
    else:
        gemini_response = get_gemini_response(user_input, conversation_history)
        new_tag = classify_intent(user_input)
        updated_intents = update_intents(user_input, gemini_response, intents, new_tag)
        save_intents(updated_intents)
        return gemini_response

if __name__ == "__main__":
    print("Chatbot is running. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        response = get_response(user_input)
        print("Bot:", response)
