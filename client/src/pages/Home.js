import React, { useState } from "react"


export const Home = () => {
    
    const [image, setImage] = useState([])
    const [prediction, setPrediction] = useState('')
    // const [loading, setLoading] = useState(false)

    const postImage = () => {
        // Set loading to true
        // setLoading(true)
        // Get recent input (uploaded image)
        let input = document.querySelector('input[type="file"]')
        // Define a FormData instance 
        let image_data = new FormData()
        // Append input to FormData object
        image_data.append('file', input.files[0])
        // Make a POST request to API
        fetch(`${process.env.REACT_APP_API}`, {
            method: "POST",
            body: image_data
        })
        // Get response
        .then(res => res.json())
        .then(data => {
            // Update prediction 
            // setLoading(false)
            setPrediction(data.class_name)
        })
    }

    const uploadImage = event => {
        setImage(URL.createObjectURL(event.target.files[0]))
    }

    return(
        <>
        <h1>Which animal is it?</h1>

        <input type="file" name="file" placeholder="Upload an image" onChange={uploadImage}/>

        <button onClick={postImage} > Predict animal </button>

        <img src={image} alt="animal" style={{width: '300px'}} />

        {/* {loading ? (<h3>Computing...</h3>) : (<h3>It's a {prediction}!</h3>)} */}
        <h1>{prediction}</h1>

        </>
    )

}
