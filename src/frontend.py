import streamlit as st
import requests

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Powerlifting Assistant")

# Sidebar for agent selection
agent_options = {
    "Router Agent": "router",
    "Search Agent": "search",
    "Chat Agent": "chat",
    "Rules Agent": "rules",
}
selected_agent = st.sidebar.selectbox("Select Agent", list(agent_options.keys()))

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message("user" if message["role"] == "user" else "assistant"):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask your question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.write(prompt)

    # Prepare the request
    request_data = {
        "agent_name": agent_options[selected_agent],
        "messages": [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ],
    }

    # Send request to API
    try:
        response = requests.post("http://localhost:8000/chat", json=request_data)
        response.raise_for_status()

        # Get the assistant's response
        assistant_response = response.json()
        if assistant_response and "messages" in assistant_response:
            last_message = assistant_response["messages"][-1]

            # Add assistant message to chat history
            st.session_state.messages.append(
                {"role": "assistant", "content": last_message["content"]}
            )

            # Display assistant message
            with st.chat_message("assistant"):
                st.write(last_message["content"])
    except Exception as e:
        st.error(f"Error communicating with the API: {str(e)}")
