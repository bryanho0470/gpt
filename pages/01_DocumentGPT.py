import streamlit as st
import time

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="💬"
)

st.title("DocumentGPT")

if "messages" not in st.session_state:
    # session_state is the dictionary
    st.session_state["messages"] = []

def send_message(message, role, save=True):
    # we call send_message() UTILITY FUNCTION which defined by users.
    with st.chat_message(role):
        st.write(message)
    if save:
        st.session_state["messages"].append({"message":message, "role": role})

for message in st.session_state["messages"]:
    send_message(message["message"], message["role"], save=False)


message = st.chat_input("Send message to AI")

#  if message mean there is something. but it doesnt work if you code "message !=""," but it is same with != None 
if message != None: 
    send_message(message, 'human')
    time.sleep(2)
    send_message(f"You said: {message}","ai")

    