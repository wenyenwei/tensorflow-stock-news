const express = require('express');
const app = express();
let data = [];
const request = require('request');
const month_length_small = ["4", "6", "9", "11"]; 
const cheerio = require('cheerio');
let result = {};
const fs = require('fs');

function requestData(){
	request
		.get(url)
		.on('response', res => {
			res.on('data', body => {
				let $ = cheerio.load(body);

				let ball;

				$($('.topStory > h2').get().reverse()).each(function(){
					ball = $(this).text().trim();
					result[mm + dd + yy] = ball;
					fs.appendFile('rnews_apple.txt', mm + dd + yy + ": \"" + ball + "\",\n", function (err) {
						if (err) throw err;
						console.log('Saved!');
					});
					console.log(mm + dd + yy, ball);
					// data.push(ball);
				});
			});
		})
		.on('error', err => { console.log("ERROR: " + err)});
}
let count = 0;
async function f() {
	
	for (let yy = 2013; yy < 2019; yy++){
		
		yy = yy.toString();

		for (let mm = 1; mm < 13; mm++){

			mm = (mm < 10) ? "0" + mm.toString() : mm.toString();
			
			let month_length = month_length_small.includes(mm) ? 30 : 31;
			
			month_length = (mm == "2" && yy == "2016") ? 29 : 28;
			
			for (let dd = 1; dd < month_length + 1; dd++){
				count += 1;

				dd = (dd < 10) ? "0" + dd.toString() : dd.toString();

				let url = 'https://www.reuters.com/finance/stocks/company-news/AAPL.OQ?date=' + mm + dd + yy; 
				
				setTimeout(() => {
					request
						.get(url)
						.on('response', res => {
							res.on('data', body => {
								let $ = cheerio.load(body);

								let ball;

								$($('.topStory > h2').get().reverse()).each(function(){
									ball = $(this).text().trim();
									result[mm + dd + yy] = ball;
									fs.appendFile('rnews_apple.txt', mm + dd + yy + ": \"" + ball + "\",\n", function (err) {
										if (err) throw err;
										console.log('Saved!');
									});
									console.log(mm + dd + yy, ball);
									// data.push(ball);
								});
							});
						})
						.on('error', err => { console.log("ERROR: " + err)});
				}, 5000 * count);
				
			}
		}
	}
}
f();





app.get('/', function (req, res) {
  // res.send(data);
});

app.listen(5000, function () {
  console.log('app listening on port 5000!');
});
