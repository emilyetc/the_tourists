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