import React, { useState } from 'react';
import './App.css'; // Create this file for styling

function App() {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (inputText.trim() === '') return;

    const newUserMessage = { sender: 'user', text: inputText };
    setMessages((prevMessages) => [...prevMessages, newUserMessage]);
    setInputText('');
    setLoading(true);

    try {
      const response = await fetch('http://127.0.0.1:5000/ask', { // Make sure this URL matches your Flask backend
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: inputText }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      const botResponse = { sender: 'bot', text: data.response };
      setMessages((prevMessages) => [...prevMessages, botResponse]);
    } catch (error) {
      console.error("Error sending message:", error);
      setMessages((prevMessages) => [...prevMessages, { sender: 'bot', text: 'Error: Could not get a response.' }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      sendMessage();
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Babji - Your Project Assistant</h1>
      </header>
      <div className="chat-container">
        <div className="messages-display">
          {messages.map((msg, index) => (
            <div key={index} className={`message ${msg.sender}`}>
              <span className="sender">{msg.sender === 'user' ? 'You: ' : 'Babji: '}</span>
              {msg.text}
            </div>
          ))}
          {loading && <div className="message bot loading">Babji is typing...</div>}
        </div>
        <div className="input-area">
          <input
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your message..."
            disabled={loading}
          />
          <button onClick={sendMessage} disabled={loading}>Send</button>
        </div>
      </div>
    </div>
  );
}

export default App;