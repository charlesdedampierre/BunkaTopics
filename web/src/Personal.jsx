function saveToPersonalSpace(content) {
  // Check if Personal.js space exists in localStorage
  let personalSpace = localStorage.getItem("Personal.js");
  if (!personalSpace) {
    personalSpace = {};
  } else {
    personalSpace = JSON.parse(personalSpace);
  }

  // Generate a unique key for the content (e.g., using a timestamp)
  const key = Date.now().toString();

  // Add the content to the Personal.js space
  personalSpace[key] = content;

  // Save the updated Personal.js space to localStorage
  localStorage.setItem("Personal.js", JSON.stringify(personalSpace));
}
