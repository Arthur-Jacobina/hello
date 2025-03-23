from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import StructuredTool
from pydantic.v1 import BaseModel, Field  # Using pydantic v1 explicitly for compatibility
from typing import List, Dict, Any, Optional, Union, Literal, Callable
from openai import OpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import sys
import json
import os
import sqlite3
import numpy as np
from typing_extensions import TypedDict  # For StateGraph compatibility
from utils import initialize
load_dotenv()

# Define standard message state class with next field
class MessagesState(TypedDict, total=False):
    """State with messages and a next field for routing."""
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]
    next: Optional[str]

class Config:
    """Configuration class for Pydantic models."""
    arbitrary_types_allowed = True
class GetUserInfoInput(BaseModel):
    username: str = Field(description="The username to retrieve information for")
    
    class Config(Config):
        pass

class SaveUserInfoInput(BaseModel):
    username: str = Field(description="The username to save information for")
    field: str = Field(description="The field to update (movie, anime, pet_preference, personality, job, color, favorite_song)")
    value: str = Field(description="The new value for the field")
    
    class Config(Config):
        pass

class FindSimilarUsersInput(BaseModel):
    username: str = Field(description="The username to find similar users for")
    top_k: int = Field(description="Number of similar users to return", default=5)
    
    class Config(Config):
        pass

# Tool implementations will be defined later
tools = []

class Embedding:
    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self.client = OpenAI()
    
    def __call__(self, text):
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding

class UserDatabase:
    def __init__(self, db_path: str = "user_info.db"):
        self.db_path = db_path
        self.embedding_model = Embedding()
        self._init_db()
    
    def _init_db(self):
        """Initialize the database with the required schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create the users table with user info and vector embeddings
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            movie TEXT,
            anime TEXT,
            pet_preference TEXT,
            personality TEXT,
            job TEXT,
            color TEXT,
            favorite_song TEXT,
            embedding_json TEXT
        )
        ''')
        conn.commit()
        conn.close()
    
    def get_user_info(self, username: str) -> Optional[Dict[str, Any]]:
        """Retrieve user information from the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        # Convert row to dictionary
        columns = ['username', 'movie', 'anime', 'pet_preference', 'personality', 'job', 'color', 'favorite_song', 'embedding_json']
        user_info = {columns[i]: row[i] for i in range(len(columns))}
        
        # Parse the embedding JSON back to a list
        user_info['embedding'] = json.loads(user_info['embedding_json'])
        del user_info['embedding_json']  # Remove the JSON string version
        
        return user_info
    
    def save_user_info(self, user_info: Dict[str, Any]) -> None:
        """Save or update user information in the database."""
        # Create concatenated text for embedding
        concat_text = " ".join([
            user_info.get('username', ''),
            user_info.get('movie', ''),
            user_info.get('anime', ''),
            user_info.get('pet_preference', ''),
            user_info.get('personality', ''),
            user_info.get('job', ''),
            user_info.get('color', ''),
            user_info.get('favorite_song', '')
        ])
        
        # Generate embedding for the concatenated text
        embedding = self.embedding_model(concat_text)
        
        # Convert embedding to JSON string for storage
        embedding_json = json.dumps(embedding)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert or replace user info
        cursor.execute('''
        INSERT OR REPLACE INTO users 
        (username, movie, anime, pet_preference, personality, job, color, favorite_song, embedding_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_info.get('username', ''),
            user_info.get('movie', ''),
            user_info.get('anime', ''),
            user_info.get('pet_preference', ''),
            user_info.get('personality', ''),
            user_info.get('job', ''),
            user_info.get('color', ''),
            user_info.get('favorite_song', ''),
            embedding_json
        ))
        
        conn.commit()
        conn.close()
    
    def find_similar_users(self, query_embedding, top_k=5):
        """Find similar users based on embedding similarity."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT username, embedding_json FROM users")
        users = cursor.fetchall()
        conn.close()
        
        if not users:
            return []
        
        # Calculate cosine similarity with all users
        similarities = []
        for username, embedding_json in users:
            user_embedding = np.array(json.loads(embedding_json))
            query_embedding_np = np.array(query_embedding)
            
            # Calculate cosine similarity
            similarity = np.dot(user_embedding, query_embedding_np) / (
                np.linalg.norm(user_embedding) * np.linalg.norm(query_embedding_np)
            )
            similarities.append((username, similarity))
        
        # Sort by similarity (highest first) and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

class IntroAgent:
    """Agent that introduces itself and gathers initial user information."""
    def __init__(self, model: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=model)
        self.system_prompt = """
        You are an assistant that helps gather information about users to personalize their experience.
        Ask questions to understand the user's personality, interests, and preferences.
        Specifically, try to learn about their:
        1. Favorite movie
        2. Favorite anime (if they watch anime)
        3. Whether they prefer dogs or cats
        4. Their personality traits
        5. Their job or occupation
        6. Favorite color
        7. Favorite song
        
        Ask questions in a conversational manner, one at a time. Be friendly and engaging.
        After gathering all information, summarize what you've learned in a structured JSON format.
        
        You have access to tools for interacting with the user database:
        
        - get_user_info: Retrieve user information from the database
        - save_user_info: Save or update user information in the database
        - find_similar_users: Find users with similar preferences
        
        You can use these tools to check if the user has any existing information or to find similar users.
        """
    
    def create_agent(self):
        """Create the intro agent."""
        return create_react_agent(
            self.llm,
            tools,  # Now includes database tools
            prompt = self.system_prompt
        )
    
    def parse_user_info(self, conversation):
        """Parse the gathered user information from the conversation."""
        # This is a simplified version. In a real app, you might want to use a more robust method.
        final_message = conversation[-1].content if conversation else ""
        
        # Simple extraction looking for JSON-like content
        try:
            # Look for JSON-like content in the final message
            start_idx = final_message.find('{')
            end_idx = final_message.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = final_message[start_idx:end_idx]
                user_info = json.loads(json_str)
                return user_info
            else:
                # If no JSON found, create a structured extraction
                user_info = {
                    "username": "",  # Will be set elsewhere
                    "movie": "",
                    "anime": "",
                    "pet_preference": "",
                    "personality": "",
                    "job": "",
                    "color": "",
                    "favorite_song": ""
                }
                
                # Simple pattern matching to extract information
                if "favorite movie" in final_message.lower():
                    # Very simple extraction, would need to be more robust in production
                    movie_parts = final_message.lower().split("favorite movie")
                    if len(movie_parts) > 1:
                        user_info["movie"] = movie_parts[1].split(".")[0].strip()
                
                # Similar extraction for other fields...
                
                return user_info
        
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "username": "",
                "movie": "",
                "anime": "",
                "pet_preference": "",
                "personality": "",
                "job": "",
                "color": "",
                "favorite_song": ""
            }

class DatabaseTools:
    """Collection of tools for interacting with the user database."""
    
    def __init__(self, user_db: UserDatabase):
        self.user_db = user_db
    
    def get_user_info(self, username: str) -> Dict[str, Any]:
        """
        Retrieve user information from the database.
        
        Args:
            username: The username to retrieve information for
            
        Returns:
            A dictionary containing user information, or an empty dict if user not found
        """
        user_info = self.user_db.get_user_info(username)
        if user_info:
            # Remove embedding from the response for cleaner output
            if 'embedding' in user_info:
                del user_info['embedding']
            return user_info
        return {"error": f"User '{username}' not found in database"}
    
    def save_user_info(self, username: str, field: str, value: str) -> Dict[str, Any]:
        """
        Update a specific field of user information in the database.
        
        Args:
            username: The username to update information for
            field: The field to update (movie, anime, pet_preference, etc.)
            value: The new value for the field
            
        Returns:
            A dictionary with the status of the operation
        """
        # Valid fields that can be updated
        valid_fields = [
            'movie', 'anime', 'pet_preference', 'personality', 
            'job', 'color', 'favorite_song'
        ]
        
        if field not in valid_fields:
            return {
                "success": False,
                "message": f"Invalid field '{field}'. Valid fields are: {', '.join(valid_fields)}"
            }
        
        # Get existing user info or create new entry
        user_info = self.user_db.get_user_info(username) or {'username': username}
        
        # Update the field
        user_info[field] = value
        
        # Save to database
        self.user_db.save_user_info(user_info)
        
        return {
            "success": True,
            "message": f"Updated {field} to '{value}' for user '{username}'"
        }
    
    def find_similar_users(self, username: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Find users with similar preferences based on vector similarity.
        
        Args:
            username: The username to find similar users for
            top_k: Number of similar users to return
            
        Returns:
            A dictionary with the list of similar users and their similarity scores
        """
        user_info = self.user_db.get_user_info(username)
        
        if not user_info or 'embedding' not in user_info:
            return {
                "success": False,
                "message": f"User '{username}' not found or has no embedding"
            }
        
        similar_users = self.user_db.find_similar_users(user_info['embedding'], top_k)
        
        # Format the results
        results = []
        for similar_username, similarity in similar_users:
            # Skip the user themselves
            if similar_username == username:
                continue
                
            # Get the similar user's info
            similar_user_info = self.user_db.get_user_info(similar_username)
            if similar_user_info and 'embedding' in similar_user_info:
                del similar_user_info['embedding']  # Remove embedding for cleaner output
                
            # Add to results with similarity score
            results.append({
                "username": similar_username,
                "similarity_score": round(similarity, 4),
                "user_info": similar_user_info
            })
        
        return {
            "success": True,
            "similar_users": results
        }

class PersonalizedChatAgent:
    """Agent that uses user information to provide personalized responses."""
    def __init__(self, model: str = "gpt-4o", user_info: Dict[str, Any] = None, user_db: UserDatabase = None):
        self.llm = ChatOpenAI(model=model)
        self.user_info = user_info or {}
        self.user_db = user_db
        self.system_prompt = self._create_system_prompt()
        self.tools = self._create_tools()
    
    def _create_system_prompt(self):
        """Create a personalized system prompt based on user information."""
        base_prompt = """You are a helpful assistant that provides personalized responses. 
        
You have access to a database of user information through tools. You can:
1. Retrieve user information with the get_user_info tool
2. Update user information with the save_user_info tool
3. Find similar users with the find_similar_users tool

Use these tools when appropriate to make your responses more personalized and helpful.
        
The user database contains the following fields for each user:
- username: The user's unique identifier
- movie: Favorite movie
- anime: Favorite anime
- pet_preference: Whether they prefer dogs or cats
- personality: Their personality traits
- job: Their job or occupation
- color: Favorite color
- favorite_song: Their favorite song

When updating information, only update one field at a time using the save_user_info tool.
"""
        
        if not self.user_info:
            return base_prompt
        
        # Add personalization based on available user info
        personalization = []
        
        if self.user_info.get('username'):
            personalization.append(f"You're speaking with {self.user_info['username']}.")
        
        if self.user_info.get('movie'):
            personalization.append(f"They enjoy movies like {self.user_info['movie']}.")
        
        if self.user_info.get('anime'):
            personalization.append(f"They like anime such as {self.user_info['anime']}.")
        
        if self.user_info.get('pet_preference'):
            personalization.append(f"They prefer {self.user_info['pet_preference']}.")
        
        if self.user_info.get('personality'):
            personalization.append(f"Their personality is described as {self.user_info['personality']}.")
        
        if self.user_info.get('job'):
            personalization.append(f"They work as {self.user_info['job']}.")
        
        if self.user_info.get('color'):
            personalization.append(f"Their favorite color is {self.user_info['color']}.")
        
        if self.user_info.get('favorite_song'):
            personalization.append(f"They enjoy listening to {self.user_info['favorite_song']}.")
        
        # Combine base prompt with personalization
        if personalization:
            return base_prompt + "\n\nCurrent user information:\n" + "\n".join(personalization) + "\n\nUse this information to make your responses more personalized and relevant."
        
        return base_prompt
    
    def _create_tools(self):
        """Create the tools for the agent."""
        if not self.user_db:
            return []
            
        db_tools = DatabaseTools(self.user_db)
        
        # Create structured tools
        get_user_info_tool = StructuredTool.from_function(
            func=db_tools.get_user_info,
            name="get_user_info",
            description="Get information about a user from the database",
            args_schema=GetUserInfoInput
        )
        
        save_user_info_tool = StructuredTool.from_function(
            func=db_tools.save_user_info,
            name="save_user_info",
            description="Save or update user information in the database",
            args_schema=SaveUserInfoInput
        )
        
        find_similar_users_tool = StructuredTool.from_function(
            func=db_tools.find_similar_users,
            name="find_similar_users",
            description="Find users with similar preferences based on vector similarity",
            args_schema=FindSimilarUsersInput
        )
        
        return [get_user_info_tool, save_user_info_tool, find_similar_users_tool]
    
    def create_agent(self):
        """Create the personalized chat agent."""
        return create_react_agent(
            self.llm,
            self.tools,
            prompt = self.system_prompt
        )

class UserAgentSystem:
    def __init__(self, model: str = "gpt-4o", user_db: UserDatabase = None):
        self.user_db = user_db or UserDatabase()
        self.llm = ChatOpenAI(model=model)
        self.router_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        self.db_tools = DatabaseTools(self.user_db)
        self.tools = self._create_tools()
        
    def _create_tools(self):
        """Create the structured tools for LangGraph."""
        get_user_info_tool = StructuredTool.from_function(
            func=self.db_tools.get_user_info,
            name="get_user_info",
            description="Get information about a user from the database",
            args_schema=GetUserInfoInput
        )
        
        save_user_info_tool = StructuredTool.from_function(
            func=self.db_tools.save_user_info,
            name="save_user_info",
            description="Save or update user information in the database",
            args_schema=SaveUserInfoInput
        )
        
        find_similar_users_tool = StructuredTool.from_function(
            func=self.db_tools.find_similar_users,
            name="find_similar_users",
            description="Find users with similar preferences based on vector similarity",
            args_schema=FindSimilarUsersInput
        )
        
        return [get_user_info_tool, save_user_info_tool, find_similar_users_tool]

    def _create_nodes(self, username: str):
        """Create the nodes for the graph workflow."""
        return {
            "router": self._router,
            "retrieval": self._create_retrieval_agent(username),
            "update": self._create_update_agent(username),
            "similarity": self._create_similarity_agent(username),
            "general": self._create_general_agent(username)
        }
    
    def _router(self, state: Dict) -> Dict:
        """Route the conversation to the appropriate node."""
        last_message = state["messages"][-1].content
        
        router_prompt = f"""
        Analyze this user request: "{last_message}"
        
        Determine the appropriate handler by responding with exactly ONE of these options:
        - "retrieval": If the user is asking about their own preferences or information stored in the database.
        - "update": If the user wants to update their preferences or information in the database.
        - "similarity": If the user wants to find similar users or recommendations based on preferences.
        - "general": If this is a general conversation not related to user data management.
        
        Output only one of these four options with no additional text.
        """
        
        response = self.router_llm.invoke(router_prompt)
        decision = response.content.strip().lower()
        
        if decision not in ["retrieval", "update", "similarity", "general"]:
            decision = "general"
            
        new_state = state.copy()
        new_state["next"] = decision
        return new_state
    
    def _create_retrieval_agent(self, username: str):
        """Create an agent specialized in retrieving user information."""
        retrieval_prompt = f"""
        You are a specialized agent for retrieving user information from the database.
        The current user is '{username}'.
        
        When asked about user preferences or information, use the get_user_info tool to retrieve it.
        Present the information in a friendly, conversational way.
        If the information isn't available, suggest the user might want to add it.
        
        Always use the get_user_info tool before responding to ensure you have the latest information.
        """
        
        retrieval_agent = create_react_agent(
            self.llm,
            self.tools,
            prompt = retrieval_prompt
        )
        
        return retrieval_agent
    
    def _create_update_agent(self, username: str):
        """Create an agent specialized in updating user information."""
        update_prompt = f"""
        You are a specialized agent for updating user information in the database.
        The current user is '{username}'.
        
        When the user wants to update their preferences or information:
        1. Use the get_user_info tool first to check current values
        2. Use the save_user_info tool to update specific fields
        3. Confirm the update was successful
        
        Remember to only update one field at a time with the save_user_info tool.
        Valid fields are: movie, anime, pet_preference, personality, job, color, favorite_song
        
        After updating, confirm the change and summarize the user's current preferences.
        """
        
        update_agent = create_react_agent(
            self.llm,
            self.tools,
            prompt = update_prompt
        )
        
        return update_agent
    
    def _create_similarity_agent(self, username: str):
        """Create an agent specialized in finding similar users."""
        similarity_prompt = f"""
        You are a specialized agent for finding users with similar preferences.
        The current user is '{username}'.
        
        When asked about similar users or recommendations:
        1. Use the get_user_info tool to check the current user's preferences
        2. Use the find_similar_users tool to find users with similar tastes
        3. Present the results in a friendly, helpful way
        
        When making recommendations, explain the similarities between users.
        If there are no similar users found, suggest the user might want to add more preference information.
        """
        
        similarity_agent = create_react_agent(
            self.llm,
            self.tools,
            prompt = similarity_prompt
        )
        
        return similarity_agent
    
    def _create_general_agent(self, username: str):
        """Create an agent for general conversation."""
        general_prompt = f"""
        You are a helpful assistant chatting with user '{username}'.
        
        You have access to the following tools for managing user preferences:
        - get_user_info: Retrieve user information from the database
        - save_user_info: Save or update user information in the database
        - find_similar_users: Find users with similar preferences
        
        Use these tools when relevant to personalize the conversation.
        Try to reference the user's preferences when appropriate to make the conversation more engaging.
        
        The database contains the following fields for each user:
        - username: The user's unique identifier
        - movie: Favorite movie
        - anime: Favorite anime
        - pet_preference: Whether they prefer dogs or cats
        - personality: Their personality traits
        - job: Their job or occupation
        - color: Favorite color
        - favorite_song: Their favorite song
        """
        
        general_agent = create_react_agent(
            self.llm,
            self.tools,
            prompt = general_prompt
        )
        
        return general_agent
    
    def build_graph(self, username: str):
        """Build the graph workflow with all the nodes."""
        nodes = self._create_nodes(username)
        
        workflow = StateGraph(MessagesState)
        
        # Add all nodes to the graph
        for node_name, node_func in nodes.items():
            workflow.add_node(node_name, node_func)
        
        # Add conditional edges from router to specialized nodes
        workflow.add_conditional_edges(
            "router",
            lambda state: state["next"],
            {
                "retrieval": "retrieval",
                "update": "update",
                "similarity": "similarity",
                "general": "general"
            }
        )
        
        # All specialized nodes end the workflow
        workflow.add_edge("retrieval", END)
        workflow.add_edge("update", END)
        workflow.add_edge("similarity", END)
        workflow.add_edge("general", END)
        
        # Set the entry point
        workflow.set_entry_point("router")
        
        return workflow.compile()
    
    def query(self, username: str, user_input: str):
        """Process a user query and return the response."""
        graph = self.build_graph(username)
        config = {"messages": [HumanMessage(content=user_input)]}
        result = graph.invoke(config)
        return result["messages"][-1].content

def get_username():
    """Get the username from the user."""
    return input("Please enter your username: ").strip()

def run_chat_mode(username: str, user_agent_system: UserAgentSystem):
    """Run the agent interactively based on user input."""
    print("Starting chat mode... Type 'exit' to end.")
    while True:
        try:
            user_input = input("\nInput: ")
            if user_input.lower() == "exit":
                break
            
            # Process the query through the graph workflow
            response = user_agent_system.query(username, user_input)
            print(f"\nAssistant: {response}")
            print("-------------------")

        except KeyboardInterrupt:
            print("Goodbye human!")
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}")
            print("An error occurred. Let's continue with a new question.")
            continue

def run_intro_conversation(intro_agent, username):
    """Run a conversation with the intro agent to gather user information."""
    print(f"Hello {username}! I'd like to get to know you better to personalize our conversations.")
    
    conversation = []
    while True:
        try:
            # Generate a response from the intro agent
            user_input = input("\nYou: ")
            if user_input.lower() == "exit":
                break
            
            # Create a new message
            message = HumanMessage(content=user_input)
            conversation.append(message)
            
            # Get agent response
            for chunk in intro_agent.stream({"messages": conversation}):
                if "agent" in chunk:
                    response = chunk["agent"]["messages"][0].content
                    print(f"Assistant: {response}")
                    conversation.append(SystemMessage(content=response))
                elif "tools" in chunk:
                    response = chunk["tools"]["messages"][0].content
                    print(f"Assistant: {response}")
                    conversation.append(SystemMessage(content=response))
            
            # Check if we've gathered enough information (simple heuristic)
            # In a real application, you'd have better logic here
            if len(conversation) >= 14:  # Approximately 7 questions and answers
                print("\nThank you for sharing that information! I'll use it to personalize our future conversations.")
                break
                
        except KeyboardInterrupt:
            print("Goodbye human!")
            sys.exit(0)
    
    return conversation

def main():
    # Initialize the user database
    user_db = UserDatabase()
    
    # Initialize the user agent system
    user_agent_system = UserAgentSystem(user_db=user_db)
    
    # Update the global tools list
    global tools
    tools = user_agent_system.tools
    
    # Get the username
    username = get_username()
    
    # Check if the user exists in the database
    user_info = user_db.get_user_info(username)
    
    if not user_info:
        initialize()
        
        # Create and run the intro agent with the database tools
        intro_agent = IntroAgent()
        intro_agent_executor = intro_agent.create_agent()
        conversation = run_intro_conversation(intro_agent_executor, username)
        
        # Parse the user info from the conversation
        user_info = intro_agent.parse_user_info(conversation)
        user_info['username'] = username
        
        # Save the user info to the database
        user_db.save_user_info(user_info)
        
        print("Thanks for sharing! Your preferences have been saved.")
    else:
        print(f"Welcome back, {username}!")
    
    # Run the chat mode with the user agent system
    run_chat_mode(username, user_agent_system)


# Add example test runs
def test_user_agent_system():
    """Run test queries against the user agent system."""
    user_db = UserDatabase()
    user_agent_system = UserAgentSystem(user_db=user_db)
    
    # Create a test user if needed
    test_user = "A"
    user_info = user_db.get_user_info(test_user)
    if not user_info:
        test_user_info = {
            "username": test_user,
            "movie": "The Matrix",
            "anime": "Attack on Titan",
            "pet_preference": "cats",
            "personality": "analytical and curious",
            "job": "software engineer",
            "color": "blue",
            "favorite_song": "Bohemian Rhapsody"
        }
        user_db.save_user_info(test_user_info)
        print(f"Created test user: {test_user}")
    
    # Sample queries covering different handler types
    test_queries = [
        "What are my favorite movies?",
        "I just watched Inception and loved it. Please update my favorite movie.",
        "Can you find users with similar taste in music as me?",
        "How's the weather today?",
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = user_agent_system.query(test_user, query)
        print(f"Response: {result}")
        print("-" * 50)

if __name__ == "__main__":
    # Comment/uncomment as needed
    main()
    # test_user_agent_system()