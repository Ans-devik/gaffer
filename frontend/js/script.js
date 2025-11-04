document.getElementById("contactForm").addEventListener("submit", e=>{
  e.preventDefault();
  const name = document.getElementById("name").value;
  const email = document.getElementById("email").value;
  const msg = document.getElementById("message").value;
  if(!name || !email || !msg) alert("Please fill all fields!");
  else alert("Message sent successfully!");
});
