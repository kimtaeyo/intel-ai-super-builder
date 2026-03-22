import './AdvancedSlider.css';
import React, { useState, useEffect } from 'react';
import { Slider } from '@mui/material';

const AdvancedSlider = ({
  label,
  description,
  value,
  onChange,
  onChangeCommitted, // Add new prop
  min,
  max,
  step = 1,
  disable = false,
}) => {
  // Ensure value is a number
  const numericValue = Number(value);

  const [inputValue, setInputValue] = useState(String(numericValue));

  useEffect(() => {
    setInputValue(String(numericValue));
  }, [numericValue]);

  const handleInputChange = e => {
    // Allow free-form input including empty string, partial numbers, etc.
    setInputValue(e.target.value);
  };

  const handleKeyDown = e => {
    if (e.key === 'Enter') {
      e.preventDefault();
      e.target.blur();
    }
  };

  const handleInputBlur = e => {
    const inputStr = e.target.value.trim();
    let newValue;

    if (inputStr === '' || isNaN(inputStr)) {
      newValue = min;
    } else {
      newValue = Number(inputStr);
      if (newValue < min) newValue = min;
      if (newValue > max) newValue = max;
    }

    setInputValue(String(newValue));
    onChange(newValue);
    if (onChangeCommitted) {
      onChangeCommitted(newValue);
    }
  };

  const handleSliderChange = (e, newValue) => {
    onChange(Number(newValue));
  };

  const handleSliderChangeCommitted = (event, newValue) => {
    if (onChangeCommitted) {
      onChangeCommitted(Number(newValue));
    }
  };

  return (
    <div className="advanced-slider-container">
      <div className="slider-container">
        <div className="slider-information">
          {description}
          <span className="slider-label">{label}</span>
        </div>
        <Slider
          className="slider"
          min={min}
          max={max}
          step={step}
          value={numericValue}
          onChange={handleSliderChange}
          onChangeCommitted={handleSliderChangeCommitted} // Add this line
          disabled={disable}
          aria-labelledby={`slider-label-${label}`}
          valueLabelDisplay="auto"
        />
      </div>
      <input
        className="slider-input"
        data-testid="slider-llm-token-settings-input"
        type="number"
        step={step}
        value={inputValue}
        onChange={handleInputChange}
        onKeyDown={handleKeyDown}
        onBlur={handleInputBlur}
        disabled={disable}
      />
    </div>
  );
};

export default AdvancedSlider;
