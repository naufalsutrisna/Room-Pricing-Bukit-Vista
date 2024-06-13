const express = require('express')
const path = require('path')
require('dotenv').config()

const SERVICE_URL = process.env.BASE_URL

const app = express()

app.use(express.json())
app.use(express.urlencoded({ extended: true }))

app.set('view engine', 'ejs')
app.set('views', path.join(__dirname, 'public/pages'))

app.use(express.static(path.join(__dirname, 'public')))

const router = express.Router()

router.get('/', async (_, res) => {
  res.render('index')
})

router.get('/predict', async (_, res) => {
  const response = await fetch(SERVICE_URL + '/properties')
  const { data } = await response.json()
  res.render('predict', { properties: data.properties })
})

router.get('/predict/new', async (_, res) => {
  res.render('new-predict')
})

router.post('/rooms', async (req, res) => {
  try {
    let propertyName = req.body.property_name.split(' ').join('%20')
    const response = await fetch(SERVICE_URL + '/rooms/' + propertyName)
    const { data } = await response.json()

    let options = data.room_id.map(
      (room) => `<option value=${room}>${room}</option>`
    )
    options.unshift(`<option>Select Room ID</option>`)

    res.send(options)
  } catch (error) {
    console.error(error)
    res.status(500).send('<option>Error get room id</option>')
  }
})

router.post('/predict', async (req, res) => {
  try {
    const options = {
      method: 'POST',
      body: JSON.stringify(req.body),
      headers: {
        'Content-Type': 'application/json',
      },
    }
    const response = await fetch(SERVICE_URL + '/predict', options)
    const { data } = await response.json()

    const html = `
      <div class="card-result">
       <p class="card-result__title">Result</p>
       <div class="card-result__body">
         <div>
           <p class="card-result__name">Prediction</p>
           <p class="card-result__data">${data.prediction}</p>
         </div>
       </div>
      </div>
    `
    res.send(html)
  } catch (error) {
    console.log(error)
  }
})

router.post('/predict/new', async (req, res) => {
  try {
    const options = {
      method: 'POST',
      body: JSON.stringify(req.body),
      headers: {
        'Content-Type': 'application/json',
      },
    }
    const response = await fetch(SERVICE_URL + '/predict/new', options)
    const { data } = await response.json()

    const html = `
      <div class="card-result">
        <p class="card-result__title">Result</p>
        <div class="card-result__body">
          <div>
            <p class="card-result__name">Prediction</p>
            <p class="card-result__data">${data.prediction}</p>
          </div>
        </div>
      </div>
    `

    res.send(html)
  } catch (error) {
    console.log(error)
  }
})

app.use(router)

app.listen(3000, () => {
  console.log('server running at http://localhost:' + 3000)
})
