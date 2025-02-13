$(document).ready(function() {
    $('#user-input').on('keypress', function(e) {
        if (e.which === 13 && $(this).val().trim() !== "") { // Enter key pressed
            const userMessage = $(this).val();
            $(this).val(''); // Clear input field

            // Append user message with animation
            $('#chat-box').append(`<div class="user-message"><strong>You:</strong> ${userMessage}</div>`);
            $('#chat-box').scrollTop($('#chat-box')[0].scrollHeight); // Scroll to bottom

            // Show loader while waiting for bot response
            $('#loader').css('visibility', 'visible');
            $('#chat-box').append('<div class="loader">...</div>');

            // Send message to backend and handle bot response
            $.get('/get', { msg: userMessage }, function(response) {
                $('#loader').css('visibility', 'hidden');
                // Append bot message with fade-in effect
                const botMessage = `<div class="bot-message"><strong>Bot:</strong> ${response.response}</div>`;
                $('#chat-box').append(botMessage);

                // Smooth scroll to the latest message
                $('#chat-box').scrollTop($('#chat-box')[0].scrollHeight);
            });
        }
    });

    // Focus on the input field after submitting a message (for smoother user interaction)
    $('#user-input').focus();
});
