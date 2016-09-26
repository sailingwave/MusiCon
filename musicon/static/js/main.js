$(document).ready(function() {
    $('#video').hide();

    var source = null;
    var n_clips = 0;

    $('#gobtn').click(function() {
        $('#input_control').hide();
        $('#video').show();

        source = new EventSource('/output/'+encodeURIComponent($("#video_url").val()));

        source.addEventListener('video', event_video, false);
    	source.addEventListener('end', event_end, false);

    	source.onerror = function(e) {
    		console.log(e);
    		source.close();
        };

    });

    var event_video = function(event){
        var data = JSON.parse(event.data);
        var video_url = data.video_url;
        console.log(video_url);

        n_clips = n_clips+1;
        $('#clips').append('<p>'+n_clips+'. </p>');
        $('#clips').append('<iframe width="360" height="270" src="'+video_url+'" frameborder="0" allowfullscreen></iframe>');
    }

    var event_end = function(event){
        source.close();
        $('#status>h4').text("Done!");
    }

    $('#stopbtn').click(function() {
        source.close();
    });
})

