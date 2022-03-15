import React, { useState, useEffect } from "react"


export const Image = () => {
    
    const [images, setImages] = useState([])
    const [imageURLs, setImageURLs] = useState([])

    useEffect(() => {
        if (imageURLs.length < 1) return
        const newImageUrls = []
        images.forEach(image => newImageUrls.push(URL.createObjectURL(image)))
        setImageURLs(newImageUrls)
    }, [images])

    function onImageChange(e) {
        setImages([...e.target.files])
    }
    
    return(
        <>
        <input type="file" multiple accept="image/*" onChange={onImageChange} />
        { imageURLs.map(imageSrc => <img src={imageSrc} alt="uploaded_animal_image"/>)}
        </>
    )

}
