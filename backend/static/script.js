const cityInput = document.getElementById('cityInput');
const suggestions = document.getElementById('suggestions');
const loadingIndicator = document.getElementById('loading-indicator');
cityInput.addEventListener('input', async () => {

  const query = cityInput.value.trim();
  if (query === '') {
    suggestions.innerHTML = '';
    return;
  }
  const response = await fetch(`/cities?query=${query}`);
  const cities = await response.json();
  suggestions.innerHTML = '';

  if (cities.length > 0) {
    const suggestionsList = document.createElement('ul');
    suggestionsList.className = "suggestions";

    cities.forEach((city) => {
      const listItem = document.createElement('li');
      listItem.textContent = city;
      listItem.className = "search_result";
      listItem.addEventListener('click', () => {
        cityInput.value = city;
        suggestions.innerHTML = '';
      });
      suggestionsList.appendChild(listItem);
    });

    suggestions.appendChild(suggestionsList);
  }
  else {

  }
});

const amenitiesList = document.getElementById('amenities-list');
const amenityItems = amenitiesList.getElementsByClassName('amenity-item');

function swapElements(list, indexA, indexB) {
  const temp = list[indexA];
  list[indexA] = list[indexB];
  list[indexB] = temp;
}

amenitiesList.addEventListener('click', (event) => {
  if (event.target.classList.contains('move-up')) {
    const currentItem = event.target.parentNode.parentNode;
    const prevItem = currentItem.previousElementSibling;
    if (prevItem) {
      swapElements(amenityItems, Array.from(amenityItems).indexOf(currentItem), Array.from(amenityItems).indexOf(prevItem));
      amenitiesList.insertBefore(currentItem, prevItem);
    }
  } else if (event.target.classList.contains('move-down')) {
    const currentItem = event.target.parentNode.parentNode;
    const nextItem = currentItem.nextElementSibling;
    if (nextItem) {
      swapElements(amenityItems, Array.from(amenityItems).indexOf(currentItem), Array.from(amenityItems).indexOf(nextItem));
      amenitiesList.insertBefore(nextItem, currentItem);
    }
  }
});
//logic for submitting form information below
function submit_form() {
  var city = document.getElementById("cityInput").value;
  var promptDescriptionHotel = document.getElementById("text_input_hotel").value;
  var promptDescriptionAttraction = document.getElementById("text_input_attractions").value;
  var rankings = [];
  var items = document.querySelectorAll(".amenity-item .amenity-name");
  items.forEach(function (item) {
    rankings.push(item.textContent);
  });
  var formData = new URLSearchParams();
  formData.append('city', city);
  formData.append('rankings', JSON.stringify(rankings));
  formData.append('promptDescriptionHotel', promptDescriptionHotel);
  formData.append('promptDescriptionAttraction', promptDescriptionAttraction);
  loadingIndicator.style.display = 'block';
  loadingIndicator.scrollIntoView({ block: 'start', behavior: 'smooth' });
  fetch("/find_places?" + formData.toString())
    .then(response => response.json())
    .then(hotelData => {
      loadingIndicator.style.display = 'none';
      displayResults(hotelData);
    })
    .catch(error => {
      console.error('Error:', error);
      loadingIndicator.style.display = 'none';
    });
}
function refine_search() {
  const resultsContainer = document.getElementById('results');
  resultsContainer.innerHTML = '';
  var city = document.getElementById("cityInput").value;
  var promptDescription = document.getElementById("text_input_hotel").value;
  var rankings = [];
  var items = document.querySelectorAll(".amenity-item .amenity-name");
  items.forEach(function (item) {
    rankings.push(item.textContent);
  });
  var formData = new URLSearchParams();
  formData.append('city', city);
  formData.append('rankings', JSON.stringify(rankings));
  formData.append('promptDescription', promptDescription);
  loadingIndicator.style.display = 'block';
  loadingIndicator.scrollIntoView({ block: 'start', behavior: 'smooth' });
  fetch("/refine_search?" + formData.toString())
    .then(response => response.json())
    .then(hotelData => {
      loadingIndicator.style.display = 'none';
      displayResults(hotelData);
    })
    .catch(error => {
      console.error('Error:', error);
      loadingIndicator.style.display = 'none';
    });
}
function handleFeedback(hotel, buttonType) {
  const feedbackData = {
    hotelReview: hotel.ratings,
    hotelName: hotel.title,
    buttonType: buttonType,
  };
  fetch('/feedback', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(feedbackData),
  })
    .then(response => {
      if (!response.ok) {
        throw new Error('No response');
      }
      return response.json();
    })
    .then(data => {
      console.log('Feedback sent successfully:', data);
    })
    .catch(error => {
      console.error('Error:', error);
    });
}
function displayResults(data) {
  const resultsContainer = document.getElementById('results');
  resultsContainer.innerHTML = '';
  const hotelsContainer = document.createElement('div');
  hotelsContainer.classList.add('scrollable-container');
  hotelsContainer.id = "hotels-container";
  // const hotelsContainer = document.getElementById('hotels-container');
  const attractionsContainer = document.createElement('div');
  attractionsContainer.classList.add('scrollable-container');
  attractionsContainer.id = "attractions-container";
  // const attractionsContainer = document.getElementById('attractions-container');
  const hotelResultTitle = document.createElement('h2');
  hotelResultTitle.textContent = "Popular hotels matching your query:";
  hotelsContainer.appendChild(hotelResultTitle);
  if (data.hasOwnProperty('Recommended Hotels')) {
    const hotelsArray = data['Recommended Hotels'];
    hotelsArray.forEach(item => {
      const itemDiv = document.createElement('div');
      itemDiv.classList.add('hotel');
      const nameElement = document.createElement('h3');
      nameElement.textContent = item.title + " (" + item.score + "% match)";
      const descript = document.createElement('p');
      descript.innerHTML = "<b> Average hotel attributes: </b>"
      const attributes = document.createElement('p');
      attributes.textContent = `Cleanliness: ${item.cleanliness.substring(0, 5)}, Location: ${item.location.substring(0, 5)}, Rooms: ${item.rooms.substring(0, 5)}, Service: ${item.service.substring(0, 5)}, Sleep quality: ${item["sleep quality"].substring(0, 5)}, Value: ${item.value.substring(0, 5)}`;
      itemDiv.appendChild(nameElement);
      itemDiv.appendChild(descript);
      itemDiv.appendChild(attributes);
      const descriptionElement = document.createElement('p');
      descriptionElement.innerHTML = `<b>A reviewer said</b>: <br> <span>${item.ratings}</span>`;
      itemDiv.appendChild(descriptionElement);

      const good_hotel_names = data.Good_Hotel_Names;
      const bad_hotel_names = data.Bad_Hotel_Names;

      const thumbsUpButton = document.createElement('button');
      thumbsUpButton.id = 'thumbsUpButton';
      thumbsUpButton.classList.add('thumbs-up');
      thumbsUpButton.textContent = '👍';
      if(!(typeof good_hotel_names === 'undefined') && good_hotel_names.includes(item.title)){
        thumbsUpButton.style.opacity = 0.9;
      }
      thumbsUpButton.addEventListener('click', function () {
        handleFeedback(item, 'thumbsUp');
        const thumbsUpStyle = window.getComputedStyle(thumbsUpButton);
        if (thumbsUpStyle.opacity == 0.5) {
          thumbsUpButton.style.opacity = 0.9;
        }
        else {
          thumbsUpButton.style.opacity = 0.5;
        }
      });
      itemDiv.appendChild(thumbsUpButton);

      const thumbsDownButton = document.createElement('button');
      thumbsDownButton.id = 'thumbsDownButton';
      thumbsDownButton.classList.add('thumbs-down');
      thumbsDownButton.textContent = '👎';
      if(!(typeof bad_hotel_names === 'undefined') && bad_hotel_names.includes(item.title)){
        thumbsDownButton.style.opacity = 0.9;
      }
      thumbsDownButton.addEventListener('click', function () {
        handleFeedback(item, 'thumbsDown');
        const thumbsDownStyle = window.getComputedStyle(thumbsDownButton);
        if (thumbsDownStyle.opacity == 0.5) {
          thumbsDownButton.style.opacity = 0.9;
        }
        else {
          thumbsDownButton.style.opacity = 0.5;
        }
      });
      itemDiv.appendChild(thumbsDownButton);

      hotelsContainer.appendChild(itemDiv);
    });
    resultsContainer.appendChild(hotelsContainer);
  }
  if (data.hasOwnProperty('Recommended Attractions')) {
    const attractionsArray = data['Recommended Attractions'];
    const attractionsResultTitle = document.createElement('h2');
    attractionsResultTitle.textContent = "Popular attractions based on your interests:";
    attractionsContainer.appendChild(attractionsResultTitle);
    attractionsArray.forEach(item => {
      const itemDiv = document.createElement('div');
      itemDiv.classList.add('attraction');

      const nameElement = document.createElement('h3');
      nameElement.textContent = item.Location_Name;
      itemDiv.appendChild(nameElement);

      const descriptionElement = document.createElement('p');
      descriptionElement.innerHTML = `A brief description: <br> <span>${item.Description}</span>`;
      itemDiv.appendChild(descriptionElement);

      attractionsContainer.appendChild(itemDiv);
    });
    resultsContainer.appendChild(attractionsContainer);
  }
  hotelsContainer.scrollIntoView({ block: 'start', behavior: 'smooth' });
}