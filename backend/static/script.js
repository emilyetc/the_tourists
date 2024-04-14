const cityInput = document.getElementById('cityInput');
const suggestions = document.getElementById('suggestions');

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
  var promptDescription = document.getElementById("text_input").value;
  var rankings = [];
  var items = document.querySelectorAll(".amenity-item .amenity-name");
  items.forEach(function (item) {
    rankings.push(item.textContent);
  });
  var formData = new URLSearchParams();
  formData.append('city', city);
  formData.append('rankings', JSON.stringify(rankings));
  formData.append('promptDescription', promptDescription);
  fetch("/find_places?" + formData.toString())
    .then(response => response.json())
    .then(hotelData => {
      displayResults(hotelData);
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
  hotelResultTitle.textContent = "You might like to stay at:";
  hotelsContainer.appendChild(hotelResultTitle);
  if (data.hasOwnProperty('Recommended Hotels')) {
    const hotelsArray = data['Recommended Hotels'];
    hotelsArray.forEach(item => {
      const itemDiv = document.createElement('div');
      itemDiv.classList.add('hotel');

      const nameElement = document.createElement('h3');
      nameElement.textContent = item.title;
      itemDiv.appendChild(nameElement);

      const descriptionElement = document.createElement('p');
      descriptionElement.innerHTML = `A reviewer said: <br> <span>${item.ratings}</span>`;
      itemDiv.appendChild(descriptionElement);

      hotelsContainer.appendChild(itemDiv);
    });
    resultsContainer.appendChild(hotelsContainer);
  }
  if (data.hasOwnProperty('Recommended Attractions')) {
    const attractionsArray = data['Recommended Attractions'];
    const attractionsResultTitle = document.createElement('h2');
    attractionsResultTitle.textContent = "You might like to vist:";
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