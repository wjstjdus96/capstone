var json_keyword = document.getElementById("keyword").value;
var keyword = json_keyword.replace(/^\[|\]$'/g, "").split(", ");

var word_list = [
  { text: keyword[0], weight: 14, url: "http://www.htmldrive.net/" },
  {
    text: keyword[1],
    weight: 12,
    url: "https://www.youtube.com/results?search_query=${keyword[1]}",
    title: "My Title",
  },
  {
    text: keyword[2],
    weight: 10,
    url: "javascript:alert('JavaScript in URL is OK!');",
  },
  { text: keyword[3], weight: 9 },
  { text: keyword[4], weight: 8 },
  { text: keyword[5], weight: 7 },
  { text: keyword[6], weight: 6 },
  { text: keyword[7], weight: 6 },
  { text: keyword[8], weight: 6 },
  { text: keyword[9], weight: 6 },
];
$(document).ready(function () {
  $("#my_favorite_latin_words").jQCloud(word_list);
});
