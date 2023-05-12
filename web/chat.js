function submitMessage() {
    var messagebox = document.getElementById('message');
    var message = messagebox.value;
    var chatlogs = document.getElementById('chatlogs');

    if (message.trim()) { // trim to prevent sending empty messages
        var lines = message.split('\n');
        var messageContainer = document.createElement('div');
        messageContainer.className = 'message';

        var icon = document.createElement('img');
        icon.src = 'human.png';
        icon.className = 'message-icon';
        messageContainer.appendChild(icon);

        // Create message text container
        var messageText = document.createElement('div');
        messageText.className = 'message-text';
        
        for (var i = 0; i < lines.length; i++) {
            var newMessage = document.createElement('p');
            if (lines[i].trim()) { // only append non-empty lines
                newMessage.textContent = lines[i];
                messageContainer.appendChild(newMessage);
            } else { // Append <br> in place .
                newMessage = document.createElement('br');
            }
            messageText.appendChild(newMessage);
        }

        messageContainer.appendChild(messageText);
        chatlogs.appendChild(messageContainer);
        messagebox.value = '';

        chatlogs.scrollTop = chatlogs.scrollHeight;
    }
}

document.getElementById('send-btn').addEventListener('click', function() {
    submitMessage()
});

document.getElementById('message').addEventListener('keydown', function(event) {
    if (event.key == 'Enter') {
        event.preventDefault();
        if (event.shiftKey || event.ctrlKey) {
            this.value += '\n'; // Add a newline if shift or ctrl is pressed.
        } else {
            submitMessage()
        }
    }
});
