$(document).ready(function() {
    $('#video').hide()

    $('#gobtn').click(function() {
        $('#video').show();

        var source = new EventSource('/output/');

        source.onmessage = function(event) {
            $('#may').text("1");
            $("#may").append("<p>hey</p>")
        }
    })

})