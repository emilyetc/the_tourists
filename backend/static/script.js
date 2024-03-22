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
  const response = fetch("/find_hotels?" + formData.toString())
}