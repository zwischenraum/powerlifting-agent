import streamlit as st
import requests

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_agent" not in st.session_state:
    st.session_state.current_agent = "router"

st.title("Powerlifting Assistant")

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
        "agent_name": st.session_state.current_agent,
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

            # Update current agent from response
            st.session_state.current_agent = assistant_response["agent_name"]
            
            # Add assistant message to chat history
            st.session_state.messages.append(
                {"role": "assistant", "content": last_message["content"]}
            )

            # Display assistant message
            with st.chat_message("assistant"):
                st.write(last_message["content"])
                
            # Display current agent (for debugging)
            st.sidebar.text(f"Current Agent: {st.session_state.current_agent}")
    except Exception as e:
        st.error(f"Error communicating with the API: {str(e)}")
