 window.alert = function() {};
 var defaultCSS = document.getElementById('bootstrap-css');

 function changeCSS(css) {
   if (css) $('head > link').filter(':first').replaceWith('<link rel="stylesheet" href="' + css + '" type="text/css" />');
   else $('head > link').filter(':first').replaceWith(defaultCSS);
 }

 $("#submit").click(function() {
   var data = $("#btn-input").val();
   //console.log(data);
   $('chat_log').append('<div class="row msg_container base_sent"><div class="col-md-10 col-xs-10"><div class="messages msg_receive"><p>' + data + '</p></div></div></div><div class="row msg_container base_receive"><div class="col-md-10 col-xs-10"><div class="messages msg_receive"><p>' + 'yoo' + '</p></div></div></div>');
   clearInput();
   $(".msg_container_base").stop().animate({
     scrollTop: $(".msg_container_base")[0].scrollHeight
   }, 1000);
 });

 function clearInput() {
   $("#myForm :input").each(function() {
     $(this).val(''); //hide form values
   });
 }

 $("#myForm").submit(function() {
   return false; //to prevent redirection to save.php
 });
