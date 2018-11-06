function submit_message(message) {
    $.post( "/send_message", {message: message}, handle_response);

    function handle_response(data) {
      // append the bot repsonse to the div
      $('.chat-container').append(`
        <div class="container-fluid">
            <div class="row">
                <div class="chat-message col-md-5 offset-md-6 bot-message">
                    ${data.message}
                </div>
                <img class="img-circle" src="static/img/robot2.gif" width=60px height=60px>
            </div>
        </div>
      `)
      
      $( "#loading" ).remove();
    }
}


$('#target').on('submit', function(e){
    e.preventDefault();
    const input_message = $('#input_message').val()
    // return if the user does not enter any text
    if (!input_message) {
      return
    }

    $('.chat-container').append(`
        <div class="container-fluid height-1">
            <div class="row">
                <img class="img-circle" src="static/img/man.jpg" width=50px height=50px>
                <div class="chat-message col-md-5 human-message">
                    ${input_message}
                </div>
            </div>
        </div>
    `)

    // loading 
    $('.chat-container').append(`
        <div class="chat-message text-center col-md-2 offset-md-10 bot-message" id="loading">
            <b>...</b>
        </div>
    `)

    // clear the text input 
    $('#input_message').val('')

    // send the message
    submit_message(input_message)
});