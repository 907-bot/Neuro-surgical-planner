"""
src/database/operations.py
Database operations for patient management.
"""

import os
from typing import Optional, List, Dict, Any
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from loguru import logger

from .models import Base, Patient, MRIStudy, SurgicalPlan, SimulationResult


class PatientDB:
    """Patient database operations."""

    def __init__(self, database_url: Optional[str] = None):
        if database_url is None:
            database_url = os.getenv(
                "DATABASE_URL",
                "postgresql://postgres:surgical_planner_dev@postgres:5432/surgical_planner"
            )
        self.engine = create_engine(database_url, pool_pre_ping=True)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self._ensure_tables()

    def _ensure_tables(self):
        """Create tables if they don't exist."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created/verified")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise

    def get_db(self) -> Session:
        """Get a database session."""
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()

    # ─── Patient Operations ───────────────────────────────────────────────────
    def create_patient(self, patient_data: Dict[str, Any]) -> Patient:
        """Create a new patient record."""
        db = self.SessionLocal()
        try:
            patient = Patient(**patient_data)
            db.add(patient)
            db.commit()
            db.refresh(patient)
            logger.info(f"Created patient: {patient.patient_id}")
            return patient
        except Exception as e:
            db.rollback()
            raise
        finally:
            db.close()

    def get_patient(self, patient_id: str) -> Optional[Patient]:
        """Get a patient by patient_id."""
        db = self.SessionLocal()
        try:
            return db.query(Patient).filter(Patient.patient_id == patient_id).first()
        finally:
            db.close()

    def list_patients(self, limit: int = 100, offset: int = 0) -> List[Patient]:
        """List all patients with pagination."""
        db = self.SessionLocal()
        try:
            return db.query(Patient).offset(offset).limit(limit).all()
        finally:
            db.close()

    def update_patient(self, patient_id: str, updates: Dict[str, Any]) -> Optional[Patient]:
        """Update a patient record."""
        db = self.SessionLocal()
        try:
            patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
            if not patient:
                return None
            for key, value in updates.items():
                if hasattr(patient, key):
                    setattr(patient, key, value)
            db.commit()
            db.refresh(patient)
            logger.info(f"Updated patient: {patient_id}")
            return patient
        except Exception as e:
            db.rollback()
            raise
        finally:
            db.close()

    def delete_patient(self, patient_id: str) -> bool:
        """Delete a patient and all related records."""
        db = self.SessionLocal()
        try:
            patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
            if not patient:
                return False
            db.delete(patient)
            db.commit()
            logger.info(f"Deleted patient: {patient_id}")
            return True
        except Exception as e:
            db.rollback()
            raise
        finally:
            db.close()

    # ─── MRI Study Operations ────────────────────────────────────────────────
    def create_study(self, study_data: Dict[str, Any]) -> MRIStudy:
        """Create a new MRI study record."""
        db = self.SessionLocal()
        try:
            study = MRIStudy(**study_data)
            db.add(study)
            db.commit()
            db.refresh(study)
            logger.info(f"Created MRI study: {study.study_id}")
            return study
        except Exception as e:
            db.rollback()
            raise
        finally:
            db.close()

    def get_studies_for_patient(self, patient_id: str) -> List[MRIStudy]:
        """Get all MRI studies for a patient."""
        db = self.SessionLocal()
        try:
            return db.query(MRIStudy).filter(MRIStudy.patient_id == patient_id).all()
        finally:
            db.close()

    # ─── Surgical Plan Operations ────────────────────────────────────────────
    def create_plan(self, plan_data: Dict[str, Any]) -> SurgicalPlan:
        """Create a new surgical plan."""
        db = self.SessionLocal()
        try:
            plan = SurgicalPlan(**plan_data)
            db.add(plan)
            db.commit()
            db.refresh(plan)
            logger.info(f"Created surgical plan: {plan.plan_id}")
            return plan
        except Exception as e:
            db.rollback()
            raise
        finally:
            db.close()

    def get_plans_for_patient(self, patient_id: str) -> List[SurgicalPlan]:
        """Get all surgical plans for a patient."""
        db = self.SessionLocal()
        try:
            return db.query(SurgicalPlan).filter(SurgicalPlan.patient_id == patient_id).all()
        finally:
            db.close()

    def update_plan_status(self, plan_id: str, status: str) -> Optional[SurgicalPlan]:
        """Update the status of a surgical plan."""
        db = self.SessionLocal()
        try:
            plan = db.query(SurgicalPlan).filter(SurgicalPlan.plan_id == plan_id).first()
            if not plan:
                return None
            plan.status = status
            db.commit()
            db.refresh(plan)
            logger.info(f"Updated plan {plan_id} status to: {status}")
            return plan
        except Exception as e:
            db.rollback()
            raise
        finally:
            db.close()

    # ─── Simulation Result Operations ────────────────────────────────────────
    def create_simulation_result(self, result_data: Dict[str, Any]) -> SimulationResult:
        """Create a new simulation result."""
        db = self.SessionLocal()
        try:
            result = SimulationResult(**result_data)
            db.add(result)
            db.commit()
            db.refresh(result)
            logger.info(f"Created simulation result for plan: {result.plan_id}")
            return result
        except Exception as e:
            db.rollback()
            raise
        finally:
            db.close()

    def get_simulations_for_plan(self, plan_id: str) -> List[SimulationResult]:
        """Get all simulation results for a plan."""
        db = self.SessionLocal()
        try:
            return db.query(SimulationResult).filter(SimulationResult.plan_id == plan_id).all()
        finally:
            db.close()
