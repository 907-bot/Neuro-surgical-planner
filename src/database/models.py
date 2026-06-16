"""
src/database/models.py
SQLAlchemy models for patient data storage.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, Text, JSON, ForeignKey
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

Base = declarative_base()


class Patient(Base):
    """Patient demographic and clinical data."""
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String(64), unique=True, nullable=False, index=True)
    name = Column(String(128), nullable=False)
    date_of_birth = Column(DateTime, nullable=True)
    gender = Column(String(16), nullable=True)
    diagnosis = Column(String(256), nullable=True)
    tumor_type = Column(String(64), nullable=True)
    tumor_location = Column(String(128), nullable=True)
    tumor_size = Column(Float, nullable=True)
    grade = Column(String(32), nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    studies = relationship("MRIStudy", back_populates="patient", cascade="all, delete-orphan")
    plans = relationship("SurgicalPlan", back_populates="patient", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "patient_id": self.patient_id,
            "name": self.name,
            "date_of_birth": self.date_of_birth.isoformat() if self.date_of_birth else None,
            "gender": self.gender,
            "diagnosis": self.diagnosis,
            "tumor_type": self.tumor_type,
            "tumor_location": self.tumor_location,
            "tumor_size": self.tumor_size,
            "grade": self.grade,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class MRIStudy(Base):
    """MRI study metadata and file references."""
    __tablename__ = "mri_studies"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String(64), ForeignKey("patients.patient_id"), nullable=False)
    study_id = Column(String(64), unique=True, nullable=False, index=True)
    study_date = Column(DateTime, nullable=True)
    modality = Column(String(32), nullable=True)  # T1, T1ce, T2, FLAIR
    file_path = Column(String(512), nullable=True)
    file_size_bytes = Column(Integer, nullable=True)
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    patient = relationship("Patient", back_populates="studies")
    plans = relationship("SurgicalPlan", back_populates="study")

    def to_dict(self):
        return {
            "id": self.id,
            "patient_id": self.patient_id,
            "study_id": self.study_id,
            "study_date": self.study_date.isoformat() if self.study_date else None,
            "modality": self.modality,
            "file_path": self.file_path,
            "file_size_bytes": self.file_size_bytes,
            "metadata_json": self.metadata_json,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class SurgicalPlan(Base):
    """Surgical plan with intervention details and outcomes."""
    __tablename__ = "surgical_plans"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String(64), ForeignKey("patients.patient_id"), nullable=False)
    study_id = Column(String(64), ForeignKey("mri_studies.study_id"), nullable=True)
    plan_id = Column(String(64), unique=True, nullable=False, index=True)
    actions = Column(JSON, nullable=False)  # List of surgical actions
    expected_recovery = Column(Float, nullable=True)
    expected_risk = Column(Float, nullable=True)
    blood_loss_ml = Column(Float, nullable=True)
    nerve_damage_prob = Column(Float, nullable=True)
    icu_days = Column(Float, nullable=True)
    confidence_interval = Column(JSON, nullable=True)
    status = Column(String(32), default="proposed")  # proposed, approved, completed, rejected
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    patient = relationship("Patient", back_populates="plans")
    study = relationship("MRIStudy", back_populates="plans")
    simulations = relationship("SimulationResult", back_populates="plan", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "patient_id": self.patient_id,
            "study_id": self.study_id,
            "plan_id": self.plan_id,
            "actions": self.actions,
            "expected_recovery": self.expected_recovery,
            "expected_risk": self.expected_risk,
            "blood_loss_ml": self.blood_loss_ml,
            "nerve_damage_prob": self.nerve_damage_prob,
            "icu_days": self.icu_days,
            "confidence_interval": self.confidence_interval,
            "status": self.status,
            "notes": self.notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class SimulationResult(Base):
    """Results from Monte-Carlo or counterfactual simulations."""
    __tablename__ = "simulation_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    plan_id = Column(String(64), ForeignKey("surgical_plans.plan_id"), nullable=False)
    simulation_type = Column(String(32), nullable=False)  # monte_carlo, counterfactual, snn
    n_simulations = Column(Integer, nullable=True)
    result_json = Column(JSON, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    plan = relationship("SurgicalPlan", back_populates="simulations")

    def to_dict(self):
        return {
            "id": self.id,
            "plan_id": self.plan_id,
            "simulation_type": self.simulation_type,
            "n_simulations": self.n_simulations,
            "result_json": self.result_json,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
